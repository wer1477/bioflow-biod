"""HTTP sidecar for the five-decision multitask model (biod-mt-v1).

Endpoints (loopback only, same conventions as the v0 sidecar):

  GET  /v1/health
  GET  /v1/version
  POST /v1/decide                 {kind:"tool"|"kb", state, role?, candidates, policy?}
  POST /v1/decide/role_select      {state}
  POST /v1/decide/relay_select     {from,to,why,body}
  POST /v1/decide/context_retention{state,groups:[{id?,seq,kind,tool,outcome,text}]}
  POST /v1/score-groups            legacy compatibility: maps 3-way labels to scores
  POST /v1/score-pairs             {task:"premise_judge"|"overclaim_flag", items:[{id,left,right,context?,unit?}]}

The model is a frozen Qwen3.5-9B text tower + LoRA + five heads, loaded from a
checkpoint produced by ``biod.multitask.train``. A single GPU forward per
request; requests are serialised with a lock.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

from biod.multitask.catalog import CatalogOption, load_catalog_snapshot
from biod.multitask.data import Sample
from biod.multitask.infer import predict_encoded
from biod.multitask.batching import InferenceQueue
from biod.multitask.collate import batch_key, encode_sample
from biod.multitask.model import MultiTaskBiodModel
from biod.multitask.render import render_context, render_relay, render_role, render_tool, render_kb
from biod.multitask.tasks import CONTEXT_LABELS, RELAY_VERDICTS, ROLES, TASK_SPECS
from biod.train.config import TrainConfig

SERVE_VERSION = "biod-mt-v2"


class Predictor:
    def __init__(
        self, checkpoint: str, device: str = "cuda", max_len: dict | None = None,
        *, max_batch_size: int = 1, batch_window_ms: float = 3, max_batch_tokens: int = 8192,
    ):
        self.checkpoint = checkpoint
        self.device = device
        with open(os.path.join(checkpoint, "model_config.json"), encoding="utf-8") as fh:
            mc = json.load(fh)
        self.base_model = mc["base_model"]
        self.head_hidden = int(mc.get("head_hidden", 1024))
        self.head_dim = int(mc.get("head_dim", 256))
        cfg = TrainConfig(base_model=self.base_model, head_hidden=self.head_hidden, head_dim=self.head_dim,
                          lora=True, gradient_checkpointing=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        from .tasks import ROUTING_TASKS

        routing_only = set(mc.get("tasks") or []) == set(ROUTING_TASKS)
        self.model = MultiTaskBiodModel(
            cfg, device=device, init_adapter=os.path.join(checkpoint, "adapter"),
            routing_only=routing_only,
        )
        self.model.load_heads(os.path.join(checkpoint, "heads.pt"))
        self.model.eval()
        self.catalog = load_catalog_snapshot()
        self.lock = threading.Lock()
        self.batch_queue = InferenceQueue(
            self._predict, max_size=max_batch_size, wait_ms=batch_window_ms,
            max_tokens=max_batch_tokens,
        ) if max_batch_size > 1 else None
        self.max_len = max_len or {t: TASK_SPECS[t].max_len for t in TASK_SPECS}
        # Warm up CUDA/FLA kernels before the first real request: the first
        # 9B forward pays ~6s of one-time cost, which would exceed the Node
        # side's 2s decision timeout.
        try:
            self.role("warmup")
            ids = list(self.catalog.tools)[:4]
            self.tool("warmup", [{"id": i} for i in ids], role="pi")
        except Exception as exc:  # warmup must never block startup
            sys.stderr.write(f"[biod.mt] warmup skipped: {exc}\n")

    def _resolve(self, c) -> CatalogOption:
        return _resolve_option(self.catalog, c)

    def _predict(self, key, encoded):
        (task, *_), policy = key
        with self.lock, torch.no_grad():
            return predict_encoded(self.model, self.tokenizer, encoded, task, dict(policy))

    def _run(self, sample: Sample, task: str, cfg: dict | None = None) -> dict:
        start = time.monotonic()
        enc = encode_sample(sample, self.tokenizer, self.max_len[task])
        if enc is None:
            raise ValueError("prompt exceeds max_len")
        encoded = time.monotonic()
        key = (batch_key(enc), tuple(sorted((cfg or {}).items())))
        if self.batch_queue is None:
            result, = self._predict(key, [enc])
            timing = {"queueMs": 0, "inferMs": round((time.monotonic() - encoded) * 1000, 2),
                      "batchSize": 1}
        else:
            result, timing = self.batch_queue.submit(key, enc, enc.prompt_len)
        return {**result, "timing": {**timing, "encodeMs": round((encoded - start) * 1000, 2)}}


    # -- public decisions ----------------------------------------------------
    def tool(self, state: str, candidates: list[dict], role: str = "", tau: float = 0.0) -> dict:
        options = [self._resolve(c) for c in candidates]
        sample = Sample(task="tool_select", id="request", role=role, state_text=state, options=options,
                        label={"id": options[0].id if options else "NO_TOOL"})
        pred = self._run(sample, "tool_select")
        chosen = pred["chosen_id"]
        if pred["confidence"] < tau:
            chosen = "NO_TOOL"
        return {
            "chosen": [chosen],
            "scores": [{"id": k, "p": v} for k, v in pred["probs"].items()],
            "confidence": pred["confidence"],
            "noToolProb": pred["no_tool_prob"],
            "candidateCount": len(options),
            "serviceTiming": pred["timing"],
        }

    def kb(self, state: str, candidates: list[dict], role: str = "", tau: float = 0.5, top_k: int = 3) -> dict:
        options = [self._resolve(c) for c in candidates]
        sample = Sample(task="kb_select", id="request", role=role, state_text=state, options=options,
                        label={"grades": {o.id: 0 for o in options}})
        pred = self._run(sample, "kb_select", {"kb_tau": tau, "kb_top_k": top_k})
        return {
            "chosen": pred["selected_ids"],
            "scores": [{"id": k, "p": v} for k, v in pred["scores"].items()],
            "confidence": max(pred["scores"].values()) if pred["scores"] else 0.0,
            "candidateCount": len(options),
            "serviceTiming": pred["timing"],
        }

    def pairs(self, task: str, items: list[dict]) -> dict:
        if task not in ("premise_judge", "overclaim_flag") or task not in self.model.tasks:
            raise ValueError(f"unsupported pair task {task!r}")
        if len(items) > 32:
            raise ValueError(f"too many items ({len(items)} > 32)")
        if not items:
            return {"task": task, "items": []}
        samples = []
        for i, item in enumerate(items):
            rid = str(item.get("id") or f"item-{i}")
            if task == "premise_judge":
                sample = Sample(task=task, id=rid, role=None, state_text="",
                                left=str(item.get("left") or ""), right=str(item.get("right") or ""),
                                context=str(item.get("context") or ""),
                                unit=str(item.get("unit") or "claim"),
                                label={"relation": "insufficient", "confidence": 1.0})
            else:
                sample = Sample(task=task, id=rid, role=None, state_text="",
                                left=str(item.get("left") or item.get("claim") or ""),
                                right=str(item.get("right") or item.get("evidence") or ""),
                                context=str(item.get("context") or ""),
                                label={"flags": [], "confidence": 1.0})
            samples.append(sample)
        encoded = []
        for sample in samples:
            enc = encode_sample(sample, self.tokenizer, self.max_len[task])
            if enc is None:
                raise ValueError(f"item too long: {sample.id}")
            encoded.append(enc)
        with self.lock, torch.no_grad():
            preds = predict_encoded(self.model, self.tokenizer, encoded, task, {})
        out = []
        for pred in preds:
            if task == "premise_judge":
                out.append({"id": pred["id"], "label": pred["relation"],
                            "confidence": pred["confidence"], "probs": pred["probs"]})
            else:
                out.append({"id": pred["id"], "label": None, "flags": pred["flags"],
                            "confidence": pred["confidence"], "probs": pred["probs"]})
        return {"task": task, "items": out}

    def role(self, state: str) -> dict:
        state = state[:2000]
        sample = Sample(task="role_select", id="request", role=None, state_text=state,
                        label={"role": ROLES[0], "confidence": 1.0})
        pred = self._run(sample, "role_select")
        return {"role": pred["role"], "confidence": pred["confidence"],
                "scores": [{"role": k, "p": v} for k, v in pred["probs"].items()]}

    def relay(self, from_role: str, to_role: str, why: str, body: str) -> dict:
        sample = Sample(task="relay_select", id="request", role=None, state_text="",
                        from_role=from_role, to_role=to_role, why=why, body=body,
                        label={"verdict": RELAY_VERDICTS[0], "confidence": 1.0, "reason_code": "target_owns"})
        pred = self._run(sample, "relay_select")
        return {"verdict": pred["verdict"], "confidence": pred["confidence"], "reasons": [],
                "scores": [{"verdict": k, "p": v} for k, v in pred["probs"].items()]}

    def context(self, state: str, groups: list[dict]) -> dict:
        normalized = []
        for i, g in enumerate(groups):
            raw_seq = g.get("seq", g.get("id", i + 1))
            try:
                seq = int(raw_seq)
            except (TypeError, ValueError):
                seq = i + 1
            normalized.append({
                "seq": seq,
                "kind": g.get("kind", "tool"),
                "tool": g.get("tool", ""),
                "outcome": g.get("outcome", "unknown"),
                "text": str(g.get("text") or g.get("description") or ""),
            })
        sample = Sample(task="context_retention", id="request", role=None, state_text=state, groups=normalized,
                        label=[{"seq": g["seq"], "label": CONTEXT_LABELS[0]} for g in normalized])
        pred = self._run(sample, "context_retention")
        return {"labels": pred["labels"]}


def _option(c: dict) -> CatalogOption:
    return CatalogOption(id=str(c["id"]), name=str(c.get("name") or c["id"]),
                         description=str(c.get("description", "")), kind=str(c.get("kind", "skill")))


def _resolve_option(catalog, c) -> CatalogOption:
    if isinstance(c, dict):
        return _option(c)
    cid = str(c)
    source = catalog.knowledge if "/" in cid and cid in catalog.knowledge else catalog.tools
    opt = source.get(cid)
    if opt is None:
        return CatalogOption(id=cid, name=cid, description="")
    return opt


def _score_from_label(label: str) -> float:
    return {"keep": 1.0, "summarise": 0.5, "drop": 0.0}.get(label, 0.5)


# Version pinning (G0.4): the health endpoint must expose what is actually loaded, so a
# candidate/incumbent pair can never be compared across two different checkpoints.
CKPT_FINGERPRINT_MODE = "sha256-tree-v1"


def checkpoint_fingerprint(path: str) -> dict:
    """Deterministic fingerprint of a checkpoint dir: sha256 over (relpath, file sha256)."""
    root = pathlib.Path(str(path)).resolve()
    if not root.exists():
        return {"ckptPath": str(root), "ckptSha256": "", "ckptFiles": 0, "ckptBytes": 0,
                "ckptFingerprintMode": CKPT_FINGERPRINT_MODE, "error": "checkpoint path missing"}
    if root.is_file():
        entries = [(root.name, root)]
    else:
        entries = sorted(((item.relative_to(root).as_posix(), item)
                          for item in root.rglob("*") if item.is_file()),
                         key=lambda pair: pair[0])
    digest = hashlib.sha256()
    total = 0
    for rel, file_path in entries:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        file_digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(4 << 20), b""):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
        total += file_path.stat().st_size
    return {
        "ckptPath": str(root),
        "ckptSha256": "sha256:" + digest.hexdigest(),
        "ckptFiles": len(entries),
        "ckptBytes": total,
        "ckptFingerprintMode": CKPT_FINGERPRINT_MODE,
    }


def _git_commit(repo_root: pathlib.Path) -> str:
    """Serving commit (best effort): the Biod checkout this process imported."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - fingerprinting must never block startup
        return ""


def _received_transport(payload: dict) -> dict:
    """Summarize the candidate cards the sidecar actually parsed from one request."""
    candidates = payload.get("candidates")
    rows = candidates if isinstance(candidates, list) else []
    descriptors = [row for row in rows if isinstance(row, dict)]
    description_chars = sum(len(str(row.get("description") or "")) for row in descriptors)
    catalog = payload.get("catalog") if isinstance(payload.get("catalog"), dict) else {}

    def _count(value, fallback):
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return fallback

    count = len(rows)
    return {
        "mode": "descriptions" if descriptors else "ids",
        "candidateCount": count,
        "descriptionChars": description_chars,
        "truncated": bool(catalog.get("truncated")),
        "catalogFingerprint": str(catalog.get("fingerprint") or ""),
        "catalogTotal": _count(catalog.get("total"), count),
        "shortlist": _count(catalog.get("shortlist"), count),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "biod-mt/" + SERVE_VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep logs small
        sys.stderr.write("[biod.mt] " + (fmt % args) + "\n")

    @property
    def predictor(self) -> Predictor:
        return self.server.predictor  # type: ignore[attr-defined]

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        if self.path == "/v1/health":
            payload = {"ok": True, "status": "ready", "modelVersion": SERVE_VERSION, "device": "cuda",
                       "optionCounts": {"tool": 0, "kb": 0}, "queueDepth": 0}
            batcher = getattr(self.predictor, "batch_queue", None)
            if batcher is not None:
                payload["queueDepth"] = batcher.depth()
            payload.update(getattr(self.server, "fingerprint", {}) or {})
            self._send(200, payload)
        elif self.path == "/v1/version":
            payload = {"ok": True, "modelVersion": SERVE_VERSION, "ckptOptionIdsHash": "", "catalogHash": "",
                       "optionIds": {"tool": [], "kb": []},
                       "supportsCandidateDescriptions": True,
                       "candidateBudget": {"maxCandidates": 64, "descriptionChars": 512,
                                           "totalDescriptionChars": 16384}}
            payload.update(getattr(self.server, "fingerprint", {}) or {})
            self._send(200, payload)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read()
            started = time.time()
            if self.path == "/v1/decide":
                kind = str(payload.get("kind", "tool"))
                if kind == "tool":
                    result = self.predictor.tool(str(payload.get("state", "")), list(payload.get("candidates") or []),
                                                 str(payload.get("role", "")),
                                                 float((payload.get("policy") or {}).get("tau", 0.0)))
                elif kind == "kb":
                    policy = payload.get("policy") or {}
                    result = self.predictor.kb(str(payload.get("state", "")), list(payload.get("candidates") or []),
                                               str(payload.get("role", "")),
                                               float(policy.get("tau", 0.5)), int(policy.get("topK", 3)))
                else:
                    self._send(400, {"error": f"unknown kind {kind!r}"}); return
            elif self.path == "/v1/decide/role_select":
                result = self.predictor.role(str(payload.get("state", "")))
            elif self.path == "/v1/decide/relay_select":
                result = self.predictor.relay(str(payload.get("from", "")), str(payload.get("to", "")),
                                              str(payload.get("why", "")), str(payload.get("body", "")))
            elif self.path == "/v1/decide/context_retention":
                result = self.predictor.context(str(payload.get("state", "")), list(payload.get("groups") or []))
            elif self.path == "/v1/score-pairs":
                result = self.predictor.pairs(str(payload.get("task") or ""), list(payload.get("items") or []))
            elif self.path == "/v1/score-groups":
                result = self.predictor.context(str(payload.get("state", "")), list(payload.get("groups") or []))
                result = {"scores": [{"id": str(x["seq"]), "score": _score_from_label(x["label"])} for x in result["labels"]]}
            else:
                self._send(404, {"error": "not found"}); return
            result["ok"] = True
            result["decisionId"] = str(payload.get("requestId") or "")
            result["latencyMs"] = round((time.time() - started) * 1000, 1)
            if self.path == "/v1/decide":
                # Echo what this process actually received. The client records this object,
                # so "descriptions reached the sidecar" is auditable rather than inferred.
                result["transport"] = _received_transport(payload)
            result["modelVersion"] = SERVE_VERSION
            self._send(200, result)
        except Exception as exc:  # fail loudly but never crash the server
            message = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ValueError) and "max_len" in str(exc):
                # The candidate cards fit the client budget but not this checkpoint's
                # encoder. Return a typed error so the runtime falls back to browsing
                # instead of treating it as an opaque 500.
                self._send(413, {"error": {"code": "PROMPT_TOO_LONG", "message": message}})
            else:
                self._send(500, {"error": message})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, predictor: Predictor):
        super().__init__(address, Handler)
        self.predictor = predictor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--max-batch-size", type=int, default=1, choices=range(1, 5))
    ap.add_argument("--batch-window-ms", type=float, default=3)
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    args = ap.parse_args(argv)
    if not 0 <= args.batch_window_ms <= 100 or args.max_batch_tokens < 1:
        ap.error("batch-window-ms must be between 0 and 100, max-batch-tokens must be positive")
    predictor = Predictor(
        args.ckpt, args.device, max_batch_size=args.max_batch_size,
        batch_window_ms=args.batch_window_ms, max_batch_tokens=args.max_batch_tokens,
    )
    server = Server((args.host, args.port), predictor)
    fingerprint = checkpoint_fingerprint(args.ckpt)
    fingerprint["serveCommit"] = _git_commit(pathlib.Path(__file__).resolve().parents[2])
    fingerprint["serveVersion"] = SERVE_VERSION
    fingerprint["loadedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    server.fingerprint = fingerprint
    print(f"[biod.mt] serving {SERVE_VERSION} on {args.host}:{args.port}", flush=True)
    print(f"[biod.mt] ckpt={fingerprint['ckptPath']} sha256={fingerprint['ckptSha256']} "
          f"files={fingerprint['ckptFiles']} bytes={fingerprint['ckptBytes']} "
          f"serveCommit={fingerprint['serveCommit']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
