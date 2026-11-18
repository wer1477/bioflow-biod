"""HTTP contract tests for :mod:`biod.serve` with an injected fake predictor.

No torch / transformers / checkpoint is needed: :class:`BiodService` takes the
predictor as a duck-typed dependency, so these tests exercise the real HTTP
layer (ThreadingHTTPServer + BaseHTTPRequestHandler) on an ephemeral port.

Run with::

    cd /home/wxf/AI/projects/bioagent/Biod
    python3 -m unittest discover -s tests -p 'test_serve.py' -v
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.client import HTTPConnection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biod.catalog import Option, catalog_hash
from biod.serve import BiodHTTPServer, BiodService, configure_offline_env, warmup_predictor

TOOL_COUNT = 61
KB_COUNT = 32
NO_TOOL_ID = "NO_TOOL"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeClock:
    """Deterministic monotonic clock shared by the service and its TTL cache."""

    def __init__(self, value: float = 1000.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class FakePredictor:
    """Duck-typed stand-in for ``BiodPredictor`` (see ``BiodService`` docstring)."""

    def __init__(self) -> None:
        self.tool_options = [
            Option(
                index=i,
                name=f"Tool {i}",
                description=f"tool {i} description",
                id=f"t{i}",
                kind="skill",
                layer="tool_l2",
                domain="fake",
            )
            for i in range(TOOL_COUNT)
        ]
        self.kb_options = [
            Option(
                index=i,
                name=f"KB {i}",
                description=f"kb {i} description",
                id=f"k{i}",
                kind="kb",
                layer="kb_l2",
                domain="fake",
            )
            for i in range(KB_COUNT)
        ]
        self.tool_canonical_ids = [o.id for o in self.tool_options] + [NO_TOOL_ID]
        self.kb_canonical_ids = [o.id for o in self.kb_options]
        self.no_tool_id = NO_TOOL_ID
        self.ckpt_dir = "/fake/ckpt/biod-fake-v0"
        self.device = "cpu"
        self.dtype = "float32"

        # Raw weights; ``predict_tool`` normalises them like a softmax would.
        self.tool_weights = default_tool_weights()
        self.kb_weights = default_kb_weights()
        self.option_weights: dict[str, float] = {}

        self.tool_calls = 0
        self.kb_calls = 0
        self.option_calls: list[int] = []
        self.option_batches: list[list[Option]] = []
        self.states: list[str] = []
        self.fail_with: BaseException | None = None

    # -- helpers ------------------------------------------------------------ #
    def _check_failure(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def _tool_scores(self) -> list[tuple[str, float]]:
        total = sum(self.tool_weights.get(oid, 0.0) for oid in self.tool_canonical_ids)
        if total <= 0.0:  # pragma: no cover - defensive
            uniform = 1.0 / len(self.tool_canonical_ids)
            return [(oid, uniform) for oid in self.tool_canonical_ids]
        return [(oid, self.tool_weights.get(oid, 0.0) / total) for oid in self.tool_canonical_ids]

    # -- predictor API ------------------------------------------------------ #
    def predict_tool(self, state: str, order: str = "canonical") -> list[tuple[str, float]]:
        self.tool_calls += 1
        self.states.append(state)
        self._check_failure()
        return self._tool_scores()

    def predict_kb(self, state: str, order: str = "canonical") -> list[tuple[str, float]]:
        self.kb_calls += 1
        self.states.append(state)
        self._check_failure()
        return [(oid, self.kb_weights.get(oid, 0.0)) for oid in self.kb_canonical_ids]

    def predict_options(
        self,
        state: str,
        options: list[Option],
        order: str = "canonical",
    ) -> list[tuple[str, float]]:
        self.option_calls.append(len(options))
        self.option_batches.append(list(options))
        self.states.append(state)
        self._check_failure()
        return [(option.id, self.option_weights.get(option.id, 0.5)) for option in options]


def default_tool_weights() -> dict[str, float]:
    """NO_TOOL dominates: 61 x 0.009 + 0.451 == 1.0."""
    weights = {oid: 0.009 for oid in (f"t{i}" for i in range(TOOL_COUNT))}
    weights[NO_TOOL_ID] = 0.451
    return weights


def default_kb_weights() -> dict[str, float]:
    return {"k0": 0.9, "k1": 0.8, "k2": 0.6, "k3": 0.2, **{f"k{i}": 0.05 for i in range(4, KB_COUNT)}}


class BlockingPredictor(FakePredictor):
    """Stalls inside the first forward so the queue depth can be observed."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict_tool(self, state: str, order: str = "canonical") -> list[tuple[str, float]]:
        self.entered.set()
        self.release.wait(timeout=10)
        return super().predict_tool(state, order=order)


# --------------------------------------------------------------------------- #
# HTTP fixture
# --------------------------------------------------------------------------- #
class ServeHTTPTestCase(unittest.TestCase):
    """Boots a real ``BiodHTTPServer`` on 127.0.0.1:0 for every test."""

    ready = True
    predictor_factory = FakePredictor
    service_kwargs: dict = {}

    def setUp(self) -> None:
        self.clock = FakeClock()
        kwargs = {
            "ckpt_dir": "/fake/ckpt/biod-fake-v0",
            "model_version": "fake-v0",
            "device": "cpu",
            "dtype": "float32",
            "clock": self.clock,
        }
        kwargs.update(self.service_kwargs)
        self.service = BiodService(**kwargs)
        self.fake = self.predictor_factory()
        if self.ready:
            self.service.set_predictor(self.fake)

        self.server = BiodHTTPServer(("127.0.0.1", 0), self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="biod-serve-test",
            daemon=True,
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    # -- client helpers ----------------------------------------------------- #
    def request(self, method: str, path: str, body: dict | None = None, raw: bytes | None = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            data = raw
            headers = {}
            if data is None and body is not None:
                data = json.dumps(body).encode("utf-8")
            if data is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            status = response.status
        finally:
            connection.close()
        if not payload:
            return status, None
        try:
            return status, json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - would be a server bug
            return status, payload

    def get_json(self, path: str):
        return self.request("GET", path)

    def post_json(self, path: str, body: dict):
        return self.request("POST", path, body=body)

    def decide(self, **body):
        return self.post_json("/v1/decide", body)

    # -- assertions --------------------------------------------------------- #
    def assert_api_error(self, status: int, body: dict, code: str) -> None:
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], code)
        self.assertIsInstance(body["error"]["message"], str)
        self.assertIsInstance(body["error"]["retryable"], bool)


# --------------------------------------------------------------------------- #
# health / version
# --------------------------------------------------------------------------- #
class TestHealthAndVersion(ServeHTTPTestCase):
    def test_health_ready(self) -> None:
        status, body = self.get_json("/v1/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["modelVersion"], "fake-v0")
        self.assertEqual(body["device"], "cpu")
        self.assertEqual(body["optionCounts"], {"tool": TOOL_COUNT, "kb": KB_COUNT})
        self.assertGreaterEqual(body["uptimeSec"], 0.0)
        self.assertEqual(body["queueDepth"], 0)
        self.assertIsNone(body["lastError"])

    def test_version_metadata(self) -> None:
        status, body = self.get_json("/v1/version")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["modelVersion"], "fake-v0")
        self.assertEqual(body["ckptDir"], "/fake/ckpt/biod-fake-v0")
        self.assertEqual(len(body["optionIds"]["tool"]), TOOL_COUNT)
        self.assertEqual(len(body["optionIds"]["kb"]), KB_COUNT)
        self.assertNotIn(NO_TOOL_ID, body["optionIds"]["tool"])
        self.assertRegex(body["ckptOptionIdsHash"], r"^[0-9a-f]{16}$")
        self.assertEqual(body["catalogHash"]["tool"], catalog_hash(self.fake.tool_options))
        self.assertEqual(body["catalogHash"]["kb"], catalog_hash(self.fake.kb_options))

    def test_version_hash_is_stable(self) -> None:
        _, first = self.get_json("/v1/version")
        _, second = self.get_json("/v1/version")
        self.assertEqual(first["ckptOptionIdsHash"], second["ckptOptionIdsHash"])

    def test_unknown_path_and_method(self) -> None:
        status, body = self.get_json("/v1/nope")
        self.assertEqual(status, 404)
        self.assert_api_error(status, body, "BAD_REQUEST")

        status, body = self.request("GET", "/v1/decide")
        self.assertEqual(status, 405)
        self.assert_api_error(status, body, "BAD_REQUEST")

        status, body = self.request("POST", "/v1/health")
        self.assertEqual(status, 405)
        self.assert_api_error(status, body, "BAD_REQUEST")


class TestNotReady(ServeHTTPTestCase):
    ready = False

    def test_health_loading(self) -> None:
        status, body = self.get_json("/v1/health")
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["status"], "loading")
        self.assertEqual(body["optionCounts"], {"tool": 0, "kb": 0})
        self.assertEqual(body["modelVersion"], "fake-v0")

    def test_version_before_load(self) -> None:
        status, body = self.get_json("/v1/version")
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["optionIds"], {"tool": [], "kb": []})
        self.assertIsNone(body["ckptOptionIdsHash"])
        self.assertEqual(body["catalogHash"], {"tool": None, "kb": None})

    def test_decide_not_ready(self) -> None:
        for payload in (
            {"kind": "tool", "state": "need a tool"},
            {"kind": "kb", "state": "need knowledge"},
        ):
            status, body = self.decide(**payload)
            self.assertEqual(status, 503)
            self.assertFalse(body["ok"])
            self.assertEqual(body["error"]["code"], "MODEL_NOT_READY")
            self.assertTrue(body["error"]["retryable"])

    def test_score_groups_not_ready(self) -> None:
        status, body = self.post_json(
            "/v1/score-groups",
            {"state": "task", "groups": [{"id": "1", "name": "n", "description": "d"}]},
        )
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "MODEL_NOT_READY")

    def test_set_predictor_flips_to_ready(self) -> None:
        self.service.set_predictor(self.fake)
        _, body = self.get_json("/v1/health")
        self.assertTrue(body["ok"])
        self.assertEqual(body["status"], "ready")
        status, decision = self.decide(kind="tool", state="need a tool")
        self.assertEqual(status, 200)
        self.assertTrue(decision["ok"])

    def test_degraded_is_not_retryable(self) -> None:
        self.service.mark_degraded("RuntimeError: no CUDA")
        status, body = self.decide(kind="tool", state="x")
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "MODEL_NOT_READY")
        self.assertFalse(body["error"]["retryable"])
        self.assertIn("no CUDA", body["error"]["message"])
        _, health = self.get_json("/v1/health")
        self.assertEqual(health["status"], "degraded")
        self.assertIn("no CUDA", health["lastError"])


# --------------------------------------------------------------------------- #
# tool decisions
# --------------------------------------------------------------------------- #
class TestToolDecision(ServeHTTPTestCase):
    def test_full_catalog_defaults_to_no_tool(self) -> None:
        status, body = self.decide(kind="tool", state="need a tool")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["kind"], "tool")
        self.assertEqual(body["modelVersion"], "fake-v0")
        self.assertEqual(body["candidateCount"], TOOL_COUNT)
        self.assertEqual(len(body["scores"]), TOOL_COUNT + 1)
        self.assertEqual(body["chosen"], [NO_TOOL_ID])
        self.assertAlmostEqual(body["confidence"], 0.451, places=6)
        self.assertAlmostEqual(body["noToolProb"], 0.451, places=6)
        self.assertEqual(body["scores"][0]["id"], NO_TOOL_ID)
        self.assertEqual(body["warnings"], [])
        self.assertEqual(len(body["decisionId"]), 12)
        self.assertIsInstance(body["latencyMs"], float)
        probabilities = [item["p"] for item in body["scores"]]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_candidates_filter_and_renormalise(self) -> None:
        # Raw: t0=0.6, t1=0.2, NO_TOOL=0.1, t2..t60=0.1 total -> sum 1.0
        self.fake.tool_weights = {"t0": 0.6, "t1": 0.2, NO_TOOL_ID: 0.1}
        self.fake.tool_weights.update({f"t{i}": 0.1 / 59 for i in range(2, TOOL_COUNT)})
        status, body = self.decide(
            kind="tool",
            state="need a specific tool",
            candidates=["t0", "t1"],
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["candidateCount"], 2)
        self.assertEqual({item["id"] for item in body["scores"]}, {"t0", "t1", NO_TOOL_ID})
        self.assertEqual(body["chosen"], ["t0"])
        self.assertAlmostEqual(body["confidence"], 0.6 / 0.9, places=6)
        self.assertAlmostEqual(body["noToolProb"], 0.1 / 0.9, places=6)
        total = sum(item["p"] for item in body["scores"])
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_no_tool_margin_flips_choice(self) -> None:
        # candidates=[t0]: t0=0.3, NO_TOOL=0.3 -> both renormalise to 0.5.
        self.fake.tool_weights = {"t0": 0.3, NO_TOOL_ID: 0.3}
        self.fake.tool_weights.update({f"t{i}": 0.4 / 60 for i in range(1, TOOL_COUNT)})

        _, zero_margin = self.decide(
            kind="tool", requestId="margin-0", state="s", candidates=["t0"], policy={"noToolMargin": 0.0}
        )
        self.assertEqual(zero_margin["chosen"], [NO_TOOL_ID])
        self.assertAlmostEqual(zero_margin["noToolProb"], 0.5, places=6)

        _, wide_margin = self.decide(
            kind="tool", requestId="margin-1", state="s", candidates=["t0"], policy={"noToolMargin": 0.1}
        )
        self.assertEqual(wide_margin["chosen"], ["t0"])
        self.assertAlmostEqual(wide_margin["confidence"], 0.5, places=6)

    def test_no_tool_always_kept_and_visible_only_counted(self) -> None:
        _, body = self.decide(kind="tool", state="s", candidates=["t5", NO_TOOL_ID])
        self.assertEqual(body["candidateCount"], 1)
        self.assertEqual({item["id"] for item in body["scores"]}, {"t5", NO_TOOL_ID})
        self.assertIn(NO_TOOL_ID, [item["id"] for item in body["scores"]])

    def test_ckpt_mismatch(self) -> None:
        status, body = self.decide(kind="tool", state="s", candidates=["t0", "bogus-tool"])
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "CKPT_MISMATCH")
        self.assertIn("bogus-tool", body["error"]["message"])
        self.assertEqual(self.fake.tool_calls, 0, "mismatched candidates must not reach the model")

    def test_kb_candidates_reject_tool_ids(self) -> None:
        status, body = self.decide(kind="kb", state="s", candidates=["t0"])
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "CKPT_MISMATCH")

    def test_empty_candidates_is_bad_request(self) -> None:
        status, body = self.decide(kind="tool", state="s", candidates=[])
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "BAD_REQUEST")

    def test_bad_kind_and_policy(self) -> None:
        status, body = self.decide(kind="domain", state="s")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "BAD_REQUEST")

        for policy in ({"tau": 2.0}, {"noToolMargin": -0.1}, {"topK": "3"}, {"tau": True}):
            status, body = self.decide(kind="tool", state="s", policy=policy)
            self.assertEqual(status, 400, policy)
            self.assertEqual(body["error"]["code"], "BAD_REQUEST")

    def test_state_truncated_and_duplicates_ignored(self) -> None:
        status, body = self.decide(kind="tool", state="x" * 2500, candidates=["t0", "t0", "t1"])
        self.assertEqual(status, 200)
        self.assertEqual(len(self.fake.states[-1]), 240)
        self.assertTrue(any("truncated" in warning for warning in body["warnings"]))
        self.assertTrue(any("duplicate" in warning for warning in body["warnings"]))
        self.assertEqual(body["candidateCount"], 2)

    def test_internal_error_is_reported(self) -> None:
        self.fake.fail_with = RuntimeError("kernel exploded")
        status, body = self.decide(kind="tool", state="s")
        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["code"], "INTERNAL")
        self.assertIn("kernel exploded", body["error"]["message"])
        _, health = self.get_json("/v1/health")
        self.assertEqual(health["status"], "ready")
        self.assertIn("kernel exploded", health["lastError"])

    def test_timeout_mapping(self) -> None:
        self.fake.fail_with = TimeoutError("took too long")
        status, body = self.decide(kind="tool", state="s")
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "TIMEOUT")
        self.assertTrue(body["error"]["retryable"])


# --------------------------------------------------------------------------- #
# knowledge-base decisions
# --------------------------------------------------------------------------- #
class TestKbDecision(ServeHTTPTestCase):
    def test_threshold_multiselect(self) -> None:
        status, body = self.decide(kind="kb", state="need knowledge", policy={"tau": 0.5, "topK": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["kind"], "kb")
        self.assertEqual(body["candidateCount"], KB_COUNT)
        self.assertEqual(body["chosen"], ["k0", "k1"])
        self.assertAlmostEqual(body["confidence"], 0.9, places=6)
        self.assertIsNone(body["noToolProb"])
        self.assertEqual(len(body["scores"]), KB_COUNT)
        self.assertEqual(body["scores"][0]["id"], "k0")

    def test_empty_chosen_when_threshold_too_high(self) -> None:
        _, body = self.decide(kind="kb", state="s", policy={"tau": 0.95, "topK": 3})
        self.assertEqual(body["chosen"], [])
        self.assertEqual(body["confidence"], 0.0)

    def test_candidates_and_topk(self) -> None:
        _, body = self.decide(kind="kb", state="s", candidates=["k1", "k2"], policy={"tau": 0.5, "topK": 5})
        self.assertEqual(body["candidateCount"], 2)
        self.assertEqual(body["chosen"], ["k1", "k2"])
        self.assertAlmostEqual(body["confidence"], 0.8, places=6)
        self.assertEqual({item["id"] for item in body["scores"]}, {"k1", "k2"})

    def test_topk_truncates(self) -> None:
        _, body = self.decide(kind="kb", state="s", policy={"tau": 0.0, "topK": 1})
        self.assertEqual(body["chosen"], ["k0"])

    def test_kb_scores_have_no_no_tool_entry(self) -> None:
        _, body = self.decide(kind="kb", state="s")
        self.assertNotIn(NO_TOOL_ID, {item["id"] for item in body["scores"]})


# --------------------------------------------------------------------------- #
# score-groups (cold-zone ranking)
# --------------------------------------------------------------------------- #
class TestScoreGroups(ServeHTTPTestCase):
    @staticmethod
    def groups(count: int) -> list[dict]:
        return [
            {"id": str(i), "name": f"tool:exec_{i}", "description": f"description {i}"}
            for i in range(count)
        ]

    def test_scores_preserve_request_order(self) -> None:
        self.fake.option_weights = {str(i): 0.1 for i in range(5)}
        self.fake.option_weights["2"] = 0.95
        status, body = self.post_json(
            "/v1/score-groups",
            {"requestId": "sg-1", "state": "task text", "groups": self.groups(5)},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual([item["id"] for item in body["scores"]], ["0", "1", "2", "3", "4"])
        scores = {item["id"]: item["score"] for item in body["scores"]}
        self.assertAlmostEqual(scores["2"], 0.95, places=6)
        self.assertAlmostEqual(scores["0"], 0.1, places=6)
        self.assertIsInstance(body["latencyMs"], float)
        self.assertEqual(body["warnings"], [])

    def test_chunks_at_16_options(self) -> None:
        status, body = self.post_json(
            "/v1/score-groups",
            {"state": "task", "groups": self.groups(32)},
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["scores"]), 32)
        self.assertEqual(self.fake.option_calls, [16, 16])

    def test_description_truncated_to_400_chars(self) -> None:
        groups = [{"id": "g0", "name": "group zero", "description": "d" * 500}]
        status, body = self.post_json("/v1/score-groups", {"state": "task", "groups": groups})
        self.assertEqual(status, 200)
        self.assertEqual(len(self.fake.option_batches[-1][0].description), 400)
        self.assertTrue(any("truncated" in warning for warning in body["warnings"]))

    def test_state_truncated(self) -> None:
        status, body = self.post_json(
            "/v1/score-groups",
            {"state": "y" * 2100, "groups": self.groups(1)},
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(self.fake.states[-1]), 400)
        self.assertTrue(any("state truncated" in warning for warning in body["warnings"]))

    def test_duplicate_group_ids_are_deduped(self) -> None:
        groups = [
            {"id": "g", "name": "first", "description": "d"},
            {"id": "g", "name": "second", "description": "d"},
            {"id": "h", "name": "third", "description": "d"},
        ]
        status, body = self.post_json("/v1/score-groups", {"state": "s", "groups": groups})
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in body["scores"]], ["g", "h"])
        self.assertTrue(any("duplicate" in warning for warning in body["warnings"]))

    def test_group_validation(self) -> None:
        for payload in (
            {"groups": []},
            {"groups": self.groups(33)},
            {"groups": [{"id": "", "name": "n", "description": "d"}]},
            {"groups": [{"id": "1", "name": "", "description": "d"}]},
            {"groups": [{"id": "1", "name": "n", "description": 5}]},
            {"groups": "not-a-list"},
        ):
            with self.subTest(payload=payload):
                status, body = self.post_json("/v1/score-groups", {"state": "s", **payload})
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["code"], "BAD_REQUEST")

    def test_idempotent_replay(self) -> None:
        body = {"requestId": "sg-replay", "state": "task", "groups": self.groups(20)}
        _, first = self.post_json("/v1/score-groups", body)
        _, second = self.post_json("/v1/score-groups", body)
        self.assertEqual(first["scores"], second["scores"])
        self.assertNotIn("cached", first)
        self.assertTrue(second["cached"])
        self.assertEqual(self.fake.option_calls, [16, 4])


# --------------------------------------------------------------------------- #
# idempotency, truncation of requests, offline env
# --------------------------------------------------------------------------- #
class TestIdempotencyAndTransport(ServeHTTPTestCase):
    def test_replay_returns_same_decision_without_forward(self) -> None:
        payload = {"requestId": "req-1", "kind": "tool", "state": "need a tool", "candidates": ["t0"]}
        _, first = self.decide(**payload)
        _, second = self.decide(**payload)
        self.assertEqual(first["decisionId"], second["decisionId"])
        self.assertEqual(first["scores"], second["scores"])
        self.assertNotIn("cached", first)
        self.assertTrue(second["cached"])
        self.assertEqual(self.fake.tool_calls, 1)

    def test_cache_expires_after_ttl(self) -> None:
        payload = {"requestId": "req-ttl", "kind": "tool", "state": "need a tool"}
        _, first = self.decide(**payload)
        self.clock.advance(30)
        _, still_cached = self.decide(**payload)
        self.assertEqual(first["decisionId"], still_cached["decisionId"])
        self.assertEqual(self.fake.tool_calls, 1)

        self.clock.advance(31)  # 61 s after the first call
        _, refreshed = self.decide(**payload)
        self.assertNotEqual(first["decisionId"], refreshed["decisionId"])
        self.assertNotIn("cached", refreshed)
        self.assertEqual(self.fake.tool_calls, 2)

    def test_request_ids_are_namespaced_per_endpoint(self) -> None:
        _, decision = self.decide(requestId="shared", kind="tool", state="s")
        _, groups = self.post_json(
            "/v1/score-groups",
            {"requestId": "shared", "state": "s", "groups": [{"id": "1", "name": "n", "description": "d"}]},
        )
        self.assertNotIn("cached", decision)
        self.assertNotIn("cached", groups)

    def test_bad_json_and_empty_body(self) -> None:
        status, body = self.request("POST", "/v1/decide", raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "BAD_REQUEST")

        status, body = self.request("POST", "/v1/decide", raw=b"")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "BAD_REQUEST")

    def test_body_must_be_object(self) -> None:
        status, body = self.post_json("/v1/decide", ["tool"])
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "BAD_REQUEST")


class TestQueueDepth(ServeHTTPTestCase):
    predictor_factory = BlockingPredictor

    def test_queue_depth_counts_in_flight_forward(self) -> None:
        self.fake.entered.clear()
        self.fake.release.clear()
        result: dict = {}

        def call() -> None:
            result["status"], result["body"] = self.decide(kind="tool", state="slow")

        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        try:
            self.assertTrue(self.fake.entered.wait(timeout=5))
            _, health = self.get_json("/v1/health")
            self.assertEqual(health["queueDepth"], 1)
        finally:
            self.fake.release.set()
            worker.join(timeout=5)
        self.assertEqual(result["status"], 200)
        _, health = self.get_json("/v1/health")
        self.assertEqual(health["queueDepth"], 0)


class TestOfflineEnv(unittest.TestCase):
    def test_defaults_and_force_flags(self) -> None:
        env = {"HF_HUB_OFFLINE": "0", "HF_HOME": "/custom/hf"}
        result = configure_offline_env(env=env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_HOME"], "/custom/hf")
        self.assertEqual(result, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/custom/hf"})

    def test_hf_home_defaults_to_repo_cache(self) -> None:
        env: dict = {}
        result = configure_offline_env(env=env, repo_root="/repo/biod")
        self.assertEqual(result["HF_HOME"], os.path.join("/repo/biod", ".hf_cache"))
        self.assertEqual(env["HF_HOME"], "/repo/biod/.hf_cache")


class TestWarmup(unittest.TestCase):
    """The loader pays the first-forward cost before publishing ``ready``."""

    def test_warmup_runs_one_forward_per_head(self) -> None:
        fake = FakePredictor()
        elapsed_ms = warmup_predictor(fake)
        self.assertEqual(fake.tool_calls, 1)
        self.assertEqual(fake.kb_calls, 1)
        self.assertIsInstance(elapsed_ms, float)
        self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_warmup_failure_is_visible_to_the_loader(self) -> None:
        fake = FakePredictor()
        fake.fail_with = RuntimeError("no CUDA device")
        with self.assertRaises(RuntimeError):
            warmup_predictor(fake)


if __name__ == "__main__":
    unittest.main(verbosity=2)
