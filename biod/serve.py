"""Resident HTTP sidecar for the Biod decision model.

Why a sidecar: the trained decision model needs one CUDA context and the full
base weights, while the agent runtime is a Node process that must be able to
time out and fail open. This module wraps a trained
:class:`biod.train.infer.BiodPredictor` in a small JSON/HTTP API.

Design constraints (contract: ``research/harness-integration/implementation-spec.md`` §2):

* stdlib only (``http.server``); the serving layer adds no dependency.
* The socket listens **before** the model is loaded. ``/v1/decide`` and
  ``/v1/score-groups`` answer ``503 MODEL_NOT_READY`` until the weights are
  ready, so a supervisor can poll ``/v1/health`` without a cold-start race.
* One GPU => every forward pass is serialised through a single lock;
  ``queueDepth`` (running + waiting) is exposed on ``/v1/health``.
* ``requestId``-keyed responses are cached for 60 s so retries are idempotent.
* The offline HuggingFace environment (``HF_HUB_OFFLINE``, ``TRANSFORMERS_OFFLINE``,
  ``HF_HOME``) is configured *before* torch/transformers is imported.
* Errors are always JSON: ``{"ok": false, "error": {"code", "message", "retryable"}}``
  with HTTP 400/503/500 and one of ``BAD_REQUEST``, ``CKPT_MISMATCH``,
  ``MODEL_NOT_READY``, ``TIMEOUT``, ``INTERNAL``.

Run::

    python -m biod.serve --host 127.0.0.1 --port 8790 \
        --ckpt artifacts/runs/biod-v0-final --device cuda --dtype bfloat16

The predictor is a duck-typed dependency: ``tests/test_serve.py`` injects a
fake through :meth:`BiodService.set_predictor`, so the HTTP contract is tested
without torch, transformers, a GPU, or a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable, Mapping, Sequence

from biod.catalog import Option, catalog_hash
from biod.render import NO_TOOL_ID

__all__ = [
    "ApiError",
    "BiodHTTPServer",
    "BiodRequestHandler",
    "BiodService",
    "configure_offline_env",
    "main",
    "warmup_predictor",
]

SERVE_VERSION = "1.0.0"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8790

DEFAULT_STATE_MAX_CHARS = 2000

# The checkpoint was trained on short states (p99 = 212 chars, max = 219) and its
# max_len is 4096 tokens. A tool prompt already spends ~3620 tokens on the 61 option
# rows, so serving must keep the state near the training distribution or the forward
# pass would reject the prompt and every decision would fail open.
TOOL_STATE_MAX_CHARS = 240
KB_STATE_MAX_CHARS = 600
GROUP_STATE_MAX_CHARS = 400
DEFAULT_MAX_GROUPS = 32
DEFAULT_GROUP_CHUNK = 16
DEFAULT_GROUP_DESCRIPTION_MAX_CHARS = 400
DEFAULT_IDEMPOTENCY_TTL_SEC = 60.0
DEFAULT_IDEMPOTENCY_MAX_ENTRIES = 512
MAX_BODY_BYTES = 256 * 1024
WARMUP_STATE = "warmup"

# Policy defaults match ``config/biod.json`` (spec §3.7). ``tau`` is not applied
# server-side for tool decisions: the caller applies ``confidence >= tau`` so
# that the fallback policy stays in one place; the value is still validated.
DEFAULT_POLICY: dict[str, dict[str, float]] = {
    "tool": {"tau": 0.5, "topK": 3, "noToolMargin": 0.1},
    "kb": {"tau": 0.5, "topK": 3, "noToolMargin": 0.1},
}

GET_ENDPOINTS = ("/v1/health", "/v1/version")
POST_ENDPOINTS = ("/v1/decide", "/v1/score-groups")

# ``<repo>/biod/serve.py`` -> ``<repo>``
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# process environment / logging
# --------------------------------------------------------------------------- #
def configure_offline_env(
    env: Mapping[str, str] | None = None,
    repo_root: str | None = None,
) -> dict[str, str]:
    """Force offline HuggingFace behaviour before torch/transformers import.

    ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are forced to ``1`` (the
    checkpoint and base model are already on disk on the serving host).
    ``HF_HOME`` is only defaulted to ``<repo>/.hf_cache`` when unset.

    Returns the three effective values, so the caller can log them.
    """
    target = os.environ if env is None else env
    target["HF_HUB_OFFLINE"] = "1"
    target["TRANSFORMERS_OFFLINE"] = "1"
    if not target.get("HF_HOME"):
        target["HF_HOME"] = os.path.join(repo_root or _REPO_ROOT, ".hf_cache")
    return {
        "HF_HUB_OFFLINE": target["HF_HUB_OFFLINE"],
        "TRANSFORMERS_OFFLINE": target["TRANSFORMERS_OFFLINE"],
        "HF_HOME": target["HF_HOME"],
    }


def _log(message: str) -> None:
    sys.stderr.write(f"[biod.serve] {message}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    """An error that maps 1:1 onto the JSON error envelope and an HTTP status."""

    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 400,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable

    def to_response(self) -> dict:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            },
        }


def _bad_request(message: str) -> ApiError:
    return ApiError("BAD_REQUEST", message, http_status=400, retryable=False)


# --------------------------------------------------------------------------- #
# request validation helpers
# --------------------------------------------------------------------------- #
def _require_mapping(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise _bad_request("request body must be a JSON object")
    return payload


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _bad_request(f"'{key}' must be a string")
    return value


def _state_text(payload: Mapping[str, Any], max_chars: int) -> tuple[str, bool]:
    """Return ``(state, truncated)``; long states are truncated, never rejected."""
    value = payload.get("state")
    if value is None:
        return "", False
    if not isinstance(value, str):
        raise _bad_request("'state' must be a string")
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _policy_number(policy: Mapping[str, Any], key: str, default: float) -> float:
    if key not in policy or policy[key] is None:
        return float(default)
    value = policy[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad_request(f"policy.{key} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise _bad_request(f"policy.{key} must be in [0, 1], got {value!r}")
    return value


def _policy_int(policy: Mapping[str, Any], key: str, default: int) -> int:
    if key not in policy or policy[key] is None:
        return int(default)
    value = policy[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bad_request(f"policy.{key} must be an integer")
    if value < 0:
        raise _bad_request(f"policy.{key} must be >= 0, got {value}")
    return value


def _parse_policy(payload: Mapping[str, Any], kind: str) -> dict[str, float]:
    policy = payload.get("policy")
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise _bad_request("'policy' must be a JSON object")
    defaults = DEFAULT_POLICY[kind]
    return {
        "tau": _policy_number(policy, "tau", defaults["tau"]),
        "topK": _policy_int(policy, "topK", defaults["topK"]),
        "noToolMargin": _policy_number(policy, "noToolMargin", defaults["noToolMargin"]),
    }


def _candidate_ids(
    payload: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> list[str] | None:
    """Validate ``candidates``; ``None`` means "the full checkpoint catalog"."""
    if "candidates" not in payload or payload["candidates"] is None:
        return None
    raw = payload["candidates"]
    if not isinstance(raw, list):
        raise _bad_request("'candidates' must be an array of ids")
    if not raw:
        raise _bad_request(
            "'candidates' must not be empty; omit the field to score the full checkpoint catalog"
        )
    ids: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise _bad_request("'candidates' must contain non-empty strings")
        ids.append(item)
    unknown = [cid for cid in ids if cid not in allowed]
    if unknown:
        shown = ", ".join(unknown[:5]) + (" …" if len(unknown) > 5 else "")
        raise ApiError(
            "CKPT_MISMATCH",
            f"candidate ids not present in checkpoint {label} options: {shown}",
            http_status=400,
            retryable=False,
        )
    return ids


def _dedupe(ids: Iterable[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    out: list[str] = []
    duplicates: list[str] = []
    for cid in ids:
        if cid in seen:
            duplicates.append(cid)
            continue
        seen.add(cid)
        out.append(cid)
    return out, duplicates


def _score_map(pairs: Iterable[Any]) -> dict[str, float]:
    """Turn ``[(id, score), ...]`` from a predictor into ``{id: score}``."""
    out: dict[str, float] = {}
    for item in pairs:
        try:
            oid, value = item
        except (TypeError, ValueError):
            raise ApiError("INTERNAL", "predictor returned malformed scores", http_status=500)
        score = float(value)
        if not math.isfinite(score):
            raise ApiError("INTERNAL", f"predictor returned a non-finite score for {oid!r}", http_status=500)
        out[str(oid)] = score
    return out


def _option_ids_hash(tool_ids: Sequence[str], kb_ids: Sequence[str]) -> str:
    """Stable 16-hex fingerprint of the checkpoint option vocabulary."""
    payload = "tool\n" + "\n".join(tool_ids) + "\nkb\n" + "\n".join(kb_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# idempotency cache
# --------------------------------------------------------------------------- #
class _IdempotencyCache:
    """Small TTL+LRU cache keyed by ``(endpoint, requestId)``."""

    def __init__(
        self,
        ttl_sec: float = DEFAULT_IDEMPOTENCY_TTL_SEC,
        max_entries: int = DEFAULT_IDEMPOTENCY_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = float(ttl_sec)
        self._max_entries = max(1, int(max_entries))
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[Any, tuple[float, dict]] = OrderedDict()

    def _prune(self) -> None:
        now = self._clock()
        while self._entries:
            _key, (stored_at, _value) = next(iter(self._entries.items()))
            if now - stored_at < self._ttl:
                break
            self._entries.popitem(last=False)

    def get(self, key: Any) -> dict | None:
        with self._lock:
            self._prune()
            entry = self._entries.get(key)
            if entry is None:
                return None
            return entry[1]

    def put(self, key: Any, value: dict) -> None:
        with self._lock:
            self._prune()
            self._entries[key] = (self._clock(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:  # pragma: no cover - diagnostics
        with self._lock:
            return len(self._entries)


# --------------------------------------------------------------------------- #
# service: model-agnostic decision logic (predictor injected)
# --------------------------------------------------------------------------- #
class BiodService:
    """Holds the predictor and implements the decision endpoints.

    The predictor must satisfy the ``BiodPredictor`` inference subset:

    * ``predict_tool(state, order="canonical") -> [(id, p)]`` (61 tools + NO_TOOL)
    * ``predict_kb(state, order="canonical") -> [(id, sigmoid)]``
    * ``predict_options(state, options, order="canonical") -> [(id, sigmoid)]``
    * ``tool_options`` / ``kb_options`` (``list[Option]``) and
      ``tool_canonical_ids`` / ``kb_canonical_ids`` (``list[str]``)
    * ``ckpt_dir``, ``device``, ``dtype``, ``no_tool_id`` attributes

    Only ``ckpt_dir``-derived metadata and the catalogs are read at
    :meth:`set_predictor` time; every forward is serialised by one lock.
    """

    def __init__(
        self,
        *,
        ckpt_dir: str | None = None,
        model_version: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        state_max_chars: int = DEFAULT_STATE_MAX_CHARS,
        max_groups: int = DEFAULT_MAX_GROUPS,
        group_chunk: int = DEFAULT_GROUP_CHUNK,
        group_description_max_chars: int = DEFAULT_GROUP_DESCRIPTION_MAX_CHARS,
        idempotency_ttl_sec: float = DEFAULT_IDEMPOTENCY_TTL_SEC,
        idempotency_max_entries: int = DEFAULT_IDEMPOTENCY_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._state_max_chars = int(state_max_chars)
        self._max_groups = int(max_groups)
        self._group_chunk = max(1, int(group_chunk))
        self._group_description_max_chars = int(group_description_max_chars)
        self._cache = _IdempotencyCache(
            ttl_sec=idempotency_ttl_sec,
            max_entries=idempotency_max_entries,
            clock=clock,
        )
        self._started_at = clock()
        self._queue_depth = 0

        self._predictor: Any | None = None
        self._status = "loading"
        self._last_error: str | None = None

        self._ckpt_dir = os.path.abspath(ckpt_dir) if ckpt_dir else ""
        self._model_version = model_version or self._version_from_ckpt(self._ckpt_dir)
        self._device = device or ""
        self._dtype = dtype or ""
        self._no_tool_id = NO_TOOL_ID
        self._tool_ids: list[str] = []
        self._kb_ids: list[str] = []
        self._tool_options: list[Option] = []
        self._kb_options: list[Option] = []
        self._ckpt_ids_hash: str | None = None
        self._tool_catalog_hash: str | None = None
        self._kb_catalog_hash: str | None = None

    # -- lifecycle ---------------------------------------------------------- #
    @staticmethod
    def _version_from_ckpt(ckpt_dir: str) -> str:
        return os.path.basename(os.path.normpath(ckpt_dir)) if ckpt_dir else ""

    @property
    def status(self) -> str:
        return self._status

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def ckpt_dir(self) -> str:
        return self._ckpt_dir

    def set_predictor(self, predictor: Any) -> None:
        """Attach a loaded predictor and publish its catalog metadata (ready)."""
        tool_options = list(getattr(predictor, "tool_options", []) or [])
        kb_options = list(getattr(predictor, "kb_options", []) or [])
        no_tool_id = str(getattr(predictor, "no_tool_id", NO_TOOL_ID) or NO_TOOL_ID)
        tool_canonical = [str(oid) for oid in (getattr(predictor, "tool_canonical_ids", []) or [])]
        kb_canonical = [str(oid) for oid in (getattr(predictor, "kb_canonical_ids", []) or [])]
        tool_ids = [oid for oid in tool_canonical if oid != no_tool_id]
        if not tool_ids:
            tool_ids = [o.id for o in tool_options]
        kb_ids = kb_canonical or [o.id for o in kb_options]

        with self._lock:
            self._no_tool_id = no_tool_id
            self._tool_ids = tool_ids
            self._kb_ids = kb_ids
            self._tool_options = tool_options
            self._kb_options = kb_options
            self._ckpt_ids_hash = _option_ids_hash(tool_ids, kb_ids) if (tool_ids or kb_ids) else None
            self._tool_catalog_hash = catalog_hash(tool_options) if tool_options else None
            self._kb_catalog_hash = catalog_hash(kb_options) if kb_options else None
            self._device = str(getattr(predictor, "device", "") or self._device)
            self._dtype = str(getattr(predictor, "dtype", "") or self._dtype)
            ckpt_dir = getattr(predictor, "ckpt_dir", "") or ""
            if ckpt_dir:
                self._ckpt_dir = os.path.abspath(str(ckpt_dir))
            if not self._model_version:
                self._model_version = self._version_from_ckpt(self._ckpt_dir)
            # Publish the predictor only after all metadata is consistent.
            self._predictor = predictor
            self._status = "ready"
            self._last_error = None

    def mark_degraded(self, error: str) -> None:
        """Record a terminal load failure (``/v1/decide`` keeps returning 503)."""
        with self._lock:
            self._status = "degraded"
            self._last_error = str(error)

    def _raise_not_ready(self) -> None:
        if self._status == "degraded":
            raise ApiError(
                "MODEL_NOT_READY",
                self._last_error or "model failed to load",
                http_status=503,
                retryable=False,
            )
        raise ApiError(
            "MODEL_NOT_READY",
            "model is still loading",
            http_status=503,
            retryable=True,
        )

    def _predictor_or_not_ready(self) -> Any:
        predictor = self._predictor
        if predictor is None:
            self._raise_not_ready()
        return predictor

    def _run_forward(self, fn: Callable[[Any], Any]) -> Any:
        """Run one forward pass under the single-GPU lock.

        ``queueDepth`` counts running *and* waiting calls, so it reflects the
        backlog a caller would experience.
        """
        self._predictor_or_not_ready()
        self._queue_depth += 1
        try:
            with self._lock:
                predictor = self._predictor
                if predictor is None:
                    self._raise_not_ready()
                try:
                    return fn(predictor)
                except ApiError:
                    raise
                except TimeoutError as exc:
                    self._last_error = f"timeout: {exc}"
                    raise ApiError(
                        "TIMEOUT",
                        f"model forward timed out: {exc}",
                        http_status=503,
                        retryable=True,
                    )
                except Exception as exc:  # noqa: BLE001 - the API boundary
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    raise ApiError(
                        "INTERNAL",
                        f"model forward failed: {type(exc).__name__}: {exc}",
                        http_status=500,
                        retryable=False,
                    )
        finally:
            self._queue_depth -= 1

    # -- endpoints ---------------------------------------------------------- #
    def health(self) -> dict:
        status = self._status
        return {
            "ok": status == "ready",
            "status": status,
            "modelVersion": self._model_version or None,
            "device": self._device or None,
            "optionCounts": {"tool": len(self._tool_ids), "kb": len(self._kb_ids)},
            "uptimeSec": round(max(0.0, self._clock() - self._started_at), 1),
            "queueDepth": int(self._queue_depth),
            "lastError": self._last_error,
        }

    def version(self) -> dict:
        return {
            "ok": self._status == "ready",
            "modelVersion": self._model_version or None,
            "ckptDir": self._ckpt_dir or None,
            "ckptOptionIdsHash": self._ckpt_ids_hash,
            "catalogHash": {"tool": self._tool_catalog_hash, "kb": self._kb_catalog_hash},
            "optionIds": {"tool": list(self._tool_ids), "kb": list(self._kb_ids)},
        }

    def decide(self, payload: Any) -> dict:
        started = time.perf_counter()
        payload = _require_mapping(payload)
        request_id = _optional_str(payload, "requestId")
        cache_key = ("decide", request_id) if request_id else None
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                response = dict(cached)
                response["cached"] = True
                return response

        kind = payload.get("kind")
        if kind not in ("tool", "kb"):
            raise _bad_request("'kind' must be 'tool' or 'kb'")
        # Fail fast on a cold/degraded model before spending time on validation.
        self._predictor_or_not_ready()

        state_cap = min(
            self._state_max_chars,
            TOOL_STATE_MAX_CHARS if kind == "tool" else KB_STATE_MAX_CHARS,
        )
        state, state_truncated = _state_text(payload, state_cap)
        warnings: list[str] = []
        if state_truncated:
            warnings.append(f"state truncated to {state_cap} chars")

        if kind == "tool":
            candidates = _candidate_ids(payload, self._tool_id_set(), "tool")
        else:
            candidates = _candidate_ids(payload, set(self._kb_ids), "kb")
        if candidates is not None:
            candidates, duplicates = _dedupe(candidates)
            if duplicates:
                warnings.append(f"ignored {len(duplicates)} duplicate candidate id(s)")
        policy = _parse_policy(payload, kind)

        if kind == "tool":
            response = self._decide_tool(state, candidates, policy, warnings, started)
        else:
            response = self._decide_kb(state, candidates, policy, warnings, started)
        if cache_key is not None:
            self._cache.put(cache_key, response)
        return response

    def score_groups(self, payload: Any) -> dict:
        started = time.perf_counter()
        payload = _require_mapping(payload)
        request_id = _optional_str(payload, "requestId")
        cache_key = ("score-groups", request_id) if request_id else None
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                response = dict(cached)
                response["cached"] = True
                return response

        self._predictor_or_not_ready()

        state_cap = min(self._state_max_chars, GROUP_STATE_MAX_CHARS)
        state, state_truncated = _state_text(payload, state_cap)
        warnings: list[str] = []
        if state_truncated:
            warnings.append(f"state truncated to {state_cap} chars")

        groups = payload.get("groups")
        if not isinstance(groups, list) or not groups:
            raise _bad_request("'groups' must be a non-empty array")
        if len(groups) > self._max_groups:
            raise _bad_request(f"'groups' has {len(groups)} entries > max {self._max_groups}")

        options: list[Option] = []
        truncated_descriptions = 0
        duplicate_groups = 0
        seen_group_ids: set[str] = set()
        for index, entry in enumerate(groups):
            if not isinstance(entry, dict):
                raise _bad_request(f"groups[{index}] must be an object")
            group_id = entry.get("id")
            name = entry.get("name")
            description = entry.get("description", "")
            if not isinstance(group_id, str) or not group_id:
                raise _bad_request(f"groups[{index}].id must be a non-empty string")
            if not isinstance(name, str) or not name.strip():
                raise _bad_request(f"groups[{index}].name must be a non-empty string")
            if description is None:
                description = ""
            if not isinstance(description, str):
                raise _bad_request(f"groups[{index}].description must be a string")
            if len(description) > self._group_description_max_chars:
                description = description[: self._group_description_max_chars]
                truncated_descriptions += 1
            if group_id in seen_group_ids:
                duplicate_groups += 1
                continue
            seen_group_ids.add(group_id)
            options.append(
                Option(
                    index=index,
                    name=name.strip(),
                    description=description.strip(),
                    id=group_id,
                    kind="kb",
                    layer="kb_l2",
                    domain="",
                )
            )
        if duplicate_groups:
            warnings.append(f"ignored {duplicate_groups} duplicate group id(s)")
        if truncated_descriptions:
            warnings.append(
                f"truncated {truncated_descriptions} group description(s) to "
                f"{self._group_description_max_chars} chars"
            )

        scores: list[dict] = []
        for offset in range(0, len(options), self._group_chunk):
            chunk = options[offset : offset + self._group_chunk]
            pairs = self._run_forward(lambda predictor, chunk=chunk: predictor.predict_options(state, chunk))
            for oid, score in _score_map(pairs).items():
                scores.append({"id": oid, "score": score})

        response = {
            "ok": True,
            "scores": scores,
            "latencyMs": round((time.perf_counter() - started) * 1000.0, 1),
            "warnings": warnings,
        }
        if cache_key is not None:
            self._cache.put(cache_key, response)
        return response

    # -- decision semantics ------------------------------------------------- #
    def _tool_id_set(self) -> set[str]:
        return set(self._tool_ids) | {self._no_tool_id}

    def _decide_tool(
        self,
        state: str,
        candidates: list[str] | None,
        policy: Mapping[str, float],
        warnings: list[str],
        started: float,
    ) -> dict:
        raw = self._run_forward(lambda predictor: predictor.predict_tool(state))
        probabilities = _score_map(raw)
        no_tool_id = self._no_tool_id

        if candidates is None:
            visible_ids = list(self._tool_ids)
        else:
            # NO_TOOL is implicit: it is always kept and never counted as visible.
            visible_ids = [cid for cid in candidates if cid != no_tool_id]
        visible = {cid: float(probabilities.get(cid, 0.0)) for cid in visible_ids}
        no_tool_prob = float(probabilities.get(no_tool_id, 0.0))

        total = sum(visible.values()) + no_tool_prob
        if total > 0.0:
            visible = {cid: value / total for cid, value in visible.items()}
            no_tool_prob = no_tool_prob / total
        else:  # pragma: no cover - defensive: a softmax over 62 slots cannot be 0
            warnings.append("degenerate score vector; fell back to a uniform distribution")
            uniform = 1.0 / (len(visible) + 1)
            visible = {cid: uniform for cid in visible}
            no_tool_prob = uniform

        margin = float(policy["noToolMargin"])
        if visible:
            best_id = min(visible.items(), key=lambda item: (-item[1], item[0]))[0]
            best_prob = visible[best_id]
        else:
            best_id = None
            best_prob = 0.0

        if best_id is None or no_tool_prob >= best_prob + margin:
            chosen = [no_tool_id]
            confidence = no_tool_prob
        else:
            chosen = [best_id]
            confidence = best_prob

        combined = dict(visible)
        combined[no_tool_id] = no_tool_prob
        ordered_scores = [
            {"id": cid, "p": prob}
            for cid, prob in sorted(combined.items(), key=lambda item: (-item[1], item[0]))
        ]
        return {
            "ok": True,
            "decisionId": uuid.uuid4().hex[:12],
            "kind": "tool",
            "modelVersion": self._model_version,
            "candidateCount": len(visible),
            "scores": ordered_scores,
            "chosen": chosen,
            "confidence": confidence,
            "noToolProb": no_tool_prob,
            "latencyMs": round((time.perf_counter() - started) * 1000.0, 1),
            "warnings": warnings,
        }

    def _decide_kb(
        self,
        state: str,
        candidates: list[str] | None,
        policy: Mapping[str, float],
        warnings: list[str],
        started: float,
    ) -> dict:
        raw = self._run_forward(lambda predictor: predictor.predict_kb(state))
        probabilities = _score_map(raw)

        if candidates is None:
            visible_ids = list(self._kb_ids)
        else:
            visible_ids = list(candidates)
        ordered_scores = [
            {"id": cid, "p": float(probabilities.get(cid, 0.0))}
            for cid in visible_ids
        ]
        ordered_scores.sort(key=lambda item: (-item["p"], item["id"]))

        tau = float(policy["tau"])
        top_k = int(policy["topK"])
        chosen = [item["id"] for item in ordered_scores if item["p"] >= tau][:top_k]
        # ``ordered_scores`` is sorted descending, so a non-empty ``chosen``
        # always starts at the highest-scoring entry.
        confidence = float(ordered_scores[0]["p"]) if chosen else 0.0
        return {
            "ok": True,
            "decisionId": uuid.uuid4().hex[:12],
            "kind": "kb",
            "modelVersion": self._model_version,
            "candidateCount": len(visible_ids),
            "scores": ordered_scores,
            "chosen": chosen,
            "confidence": confidence,
            "noToolProb": None,
            "latencyMs": round((time.perf_counter() - started) * 1000.0, 1),
            "warnings": warnings,
        }


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
class BiodRequestHandler(BaseHTTPRequestHandler):
    """JSON router for the four endpoints; all errors keep the JSON envelope."""

    protocol_version = "HTTP/1.1"
    server_version = "biod-serve/" + SERVE_VERSION
    sys_version = ""

    def version_string(self) -> str:  # pragma: no cover - header cosmetics
        return self.server_version

    # -- routing ------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self._request_path()
        try:
            if path == "/v1/health":
                response = self.server.service.health()
            elif path == "/v1/version":
                response = self.server.service.version()
            elif path in POST_ENDPOINTS:
                raise ApiError("BAD_REQUEST", f"{path} only supports POST", http_status=405)
            else:
                raise ApiError("BAD_REQUEST", f"unknown path: {path}", http_status=404)
        except ApiError as exc:
            self._send_json(exc.http_status, exc.to_response(), close=True)
            return
        except Exception as exc:  # noqa: BLE001 - the HTTP boundary
            self._send_json(
                500,
                ApiError("INTERNAL", f"{type(exc).__name__}: {exc}", http_status=500).to_response(),
                close=True,
            )
            return
        self._send_json(200, response)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self._request_path()
        try:
            if path == "/v1/decide":
                response = self.server.service.decide(self._read_json())
            elif path == "/v1/score-groups":
                response = self.server.service.score_groups(self._read_json())
            elif path in GET_ENDPOINTS:
                raise ApiError("BAD_REQUEST", f"{path} only supports GET", http_status=405)
            else:
                raise ApiError("BAD_REQUEST", f"unknown path: {path}", http_status=404)
        except ApiError as exc:
            self._send_json(exc.http_status, exc.to_response(), close=True)
            return
        except Exception as exc:  # noqa: BLE001 - the HTTP boundary
            self._send_json(
                500,
                ApiError("INTERNAL", f"{type(exc).__name__}: {exc}", http_status=500).to_response(),
                close=True,
            )
            return
        self._send_json(200, response)

    # -- helpers ------------------------------------------------------------ #
    def _request_path(self) -> str:
        path = self.path.split("?", 1)[0]
        if len(path) > 1:
            path = path.rstrip("/")
        return path or "/"

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise _bad_request("Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError:
            raise _bad_request("invalid Content-Length header")
        if length <= 0:
            raise _bad_request("request body is empty")
        if length > MAX_BODY_BYTES:
            raise _bad_request(f"request body exceeds {MAX_BODY_BYTES} bytes")
        body = self.rfile.read(length)
        if len(body) != length:
            raise _bad_request("request body is shorter than Content-Length")
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _bad_request(f"invalid JSON body: {exc}")

    def _send_json(self, status: int, body: Mapping[str, Any], close: bool = False) -> None:
        payload = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if close:
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):  # client hung up on timeout
            self.close_connection = True

    def log_message(self, fmt: str, *args: Any) -> None:
        _log(f"{self.address_string()} {fmt % args}")


class BiodHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` carrying the :class:`BiodService` instance."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], service: BiodService) -> None:
        self.service = service
        super().__init__(server_address, BiodRequestHandler)


# --------------------------------------------------------------------------- #
# CLI / startup
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m biod.serve",
        description="Resident HTTP sidecar for the Biod decision model.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port (default: %(default)s)")
    parser.add_argument("--ckpt", required=True, help="checkpoint run dir (biod_config.json + heads.pt)")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: checkpoint / auto)")
    parser.add_argument("--dtype", default=None, help="bfloat16|float16|float32 (default: checkpoint)")
    parser.add_argument(
        "--model-version",
        default=None,
        help="version string reported by /v1/health and /v1/version (default: basename of --ckpt)",
    )
    return parser


def warmup_predictor(predictor: Any) -> float:
    """Run one throw-away forward per head; return the elapsed milliseconds.

    In a fresh process the first ``predict_tool`` costs ~7 s and the first
    ``predict_kb`` ~3 s (CUDA context + kernel selection on the long option
    prompts); afterwards they take ~0.42 s / ~0.23 s. :func:`_load_model` calls
    this **before** publishing the predictor, so the first real client decision
    already runs at steady-state latency and fits a 2 s request timeout.
    """
    started = time.perf_counter()
    predictor.predict_tool(WARMUP_STATE)
    predictor.predict_kb(WARMUP_STATE)
    return (time.perf_counter() - started) * 1000.0


def _load_model(service: BiodService, ckpt_dir: str, device: str | None, dtype: str | None) -> None:
    """Background loader: configure offline env, import torch late, publish."""
    env = configure_offline_env()
    started = time.perf_counter()
    _log(
        f"loading checkpoint {ckpt_dir!r} (device={device or 'auto'}, dtype={dtype or 'auto'}, "
        f"HF_HOME={env['HF_HOME']})"
    )
    try:
        from biod.train.infer import BiodPredictor  # deferred: pulls in torch/transformers

        predictor = BiodPredictor(ckpt_dir, device=device, dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - stay up in `degraded` so /health explains
        service.mark_degraded(f"{type(exc).__name__}: {exc}")
        _log(f"model load FAILED after {time.perf_counter() - started:.1f}s: {type(exc).__name__}: {exc}")
        return

    # First forward of each head costs seconds (CUDA/kernel init); pay it here,
    # while /v1/* still answers MODEL_NOT_READY, so the first real decision runs
    # at steady-state latency. A warmup failure is not fatal: the API stays up
    # and the caller fails open on the (slow) first call.
    try:
        warmup_ms = warmup_predictor(predictor)
        _log(f"warmup forward passes done in {warmup_ms / 1000.0:.1f}s")
    except Exception as exc:  # noqa: BLE001 - warmup is an optimisation, not a contract
        _log(f"warmup FAILED (serving anyway, first call may be slow): {type(exc).__name__}: {exc}")

    service.set_predictor(predictor)
    counts = service.health()["optionCounts"]
    _log(
        f"model ready in {time.perf_counter() - started:.1f}s "
        f"(tool={counts['tool']}, kb={counts['kb']}): {predictor!r}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Offline env must be in place before the loader thread imports transformers.
    configure_offline_env()

    service = BiodService(
        ckpt_dir=args.ckpt,
        model_version=args.model_version,
        device=args.device,
        dtype=args.dtype,
    )
    try:
        server = BiodHTTPServer((args.host, args.port), service)
    except OSError as exc:
        _log(f"failed to bind {args.host}:{args.port}: {exc}")
        return 2

    bound_host, bound_port = server.server_address[0], server.server_address[1]
    _log(
        f"listening on http://{bound_host}:{bound_port} "
        f"(ckpt={service.ckpt_dir}, modelVersion={service.model_version or 'unknown'})"
    )
    loader = threading.Thread(
        target=_load_model,
        args=(service, args.ckpt, args.device, args.dtype),
        name="biod-loader",
        daemon=True,
    )
    loader.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("interrupted, shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
