"""Inference / decoding for the five heads.

Kept separate from training so evaluation and the sidecar share exactly one
decode path. Batches are built by ``batch_key`` (task, role, slot count); a
single sample is a batch of one.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import torch

from .collate import Encoded, batch_key, encode_sample, pad_batch
from .data import Sample
from .tasks import CONTEXT_LABELS, OVERCLAIM_FLAGS, PREMISE_LABELS, RELAY_VERDICTS, ROLES

DEFAULT_MAX_BATCH = {
    "tool_select": 1,
    "kb_select": 4,
    "context_retention": 8,
    "role_select": 16,
    "relay_select": 16,
    "premise_judge": 16,
    "overclaim_flag": 16,
}


def _softmax(values: Sequence[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [v / s for v in exps]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _batches(encoded: Sequence[Encoded], task: str) -> Iterable[Sequence[Encoded]]:
    by_key = defaultdict(list)
    for enc in encoded:
        by_key[batch_key(enc)].append(enc)
    size = DEFAULT_MAX_BATCH.get(task, 8)
    for key in sorted(by_key, key=str):
        group = by_key[key]
        for i in range(0, len(group), size):
            yield group[i:i + size]


def _predict_batch(model, tokenizer, batch: Sequence[Encoded], task: str, cfg: dict) -> list[dict]:
    tensors = pad_batch(batch, tokenizer.pad_token_id, device="cuda")
    with torch.no_grad():
        out = model(tensors)
        logits = out["logits"].float().cpu()
    results: list[dict] = []
    if task == "tool_select":
        probs = logits.softmax(-1)
        for i, enc in enumerate(batch):
            p = probs[i].tolist()
            ids = enc.option_ids
            no_tool = p[-1]
            option_probs = p[:-1]
            ranked = [oid for _, oid in sorted(zip(option_probs, ids[:-1]), reverse=True)]
            best = max(range(len(p)), key=lambda j: p[j])
            chosen = ids[best]
            results.append({
                "id": enc.record_id,
                "chosen_id": chosen,
                "confidence": p[best],
                "no_tool_prob": no_tool,
                "probs": {oid: p[j] for j, oid in enumerate(ids)},
                "ranked_ids": ranked,
                "meta": enc.meta.get("sample", {}),
            })
    elif task == "kb_select":
        probs = logits.sigmoid()
        tau = float(cfg.get("kb_tau", 0.5))
        top_k = int(cfg.get("kb_top_k", 3))
        for i, enc in enumerate(batch):
            p = probs[i].tolist()
            scores = {oid: p[j] for j, oid in enumerate(enc.option_ids)}
            selected = [oid for oid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if scores[oid] >= tau][:top_k]
            results.append({
                "id": enc.record_id,
                "scores": scores,
                "selected_ids": selected,
                "meta": enc.meta.get("sample", {}),
            })
    elif task == "context_retention":
        probs = logits.softmax(-1)
        for i, enc in enumerate(batch):
            p = probs[i].tolist()
            seqs = enc.meta.get("group_seqs", [])
            labels = []
            for j, seq in enumerate(seqs):
                k = max(range(3), key=lambda x: p[j][x])
                labels.append({"seq": seq, "label": CONTEXT_LABELS[k], "confidence": p[j][k]})
            results.append({"id": enc.record_id, "labels": labels, "meta": enc.meta.get("sample", {})})
    elif task == "role_select":
        probs = logits.softmax(-1)
        for i, enc in enumerate(batch):
            p = probs[i].tolist()
            k = max(range(5), key=lambda x: p[x])
            results.append({
                "id": enc.record_id, "role": ROLES[k], "confidence": p[k],
                "probs": {r: p[j] for j, r in enumerate(ROLES)}, "meta": enc.meta.get("sample", {}),
            })
    elif task == "relay_select":
        probs = logits.softmax(-1)
        for i, enc in enumerate(batch):
            p = probs[i].tolist()
            k = max(range(3), key=lambda x: p[x])
            results.append({
                "id": enc.record_id, "verdict": RELAY_VERDICTS[k], "confidence": p[k],
                "probs": {v: p[j] for j, v in enumerate(RELAY_VERDICTS)}, "meta": enc.meta.get("sample", {}),
            })
    elif task == "premise_judge":
        probs = logits.softmax(-1)
        for i, enc in enumerate(batch):
            p = probs[i].tolist()
            k = max(range(len(PREMISE_LABELS)), key=lambda x: p[x])
            results.append({
                "id": enc.record_id, "relation": PREMISE_LABELS[k], "confidence": p[k],
                "probs": {label: p[j] for j, label in enumerate(PREMISE_LABELS)},
                "meta": enc.meta.get("sample", {}),
            })
    elif task == "overclaim_flag":
        probs = logits.sigmoid()
        for i, enc in enumerate(batch):
            p = probs[i].tolist()
            flags = [f for j, f in enumerate(OVERCLAIM_FLAGS) if p[j] >= 0.5]
            results.append({
                "id": enc.record_id, "flags": flags, "confidence": max(p) if p else 0.0,
                "probs": {f: p[j] for j, f in enumerate(OVERCLAIM_FLAGS)},
                "meta": enc.meta.get("sample", {}),
            })
    else:
        raise ValueError(f"unknown task {task!r}")
    return results


def predict_encoded(model, tokenizer, encoded: Sequence[Encoded], task: str, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or {}
    model.eval()
    out: list[dict] = []
    for batch in _batches(encoded, task):
        out.extend(_predict_batch(model, tokenizer, batch, task, cfg))
    return out


def predict_samples(
    model,
    tokenizer,
    samples: Sequence[Sample],
    task: str,
    max_len: int,
    cfg: dict | None = None,
) -> list[dict]:
    encoded = []
    skipped = []
    for sample in samples:
        enc = encode_sample(sample, tokenizer, max_len)
        if enc is None:
            skipped.append(sample.id)
        else:
            encoded.append(enc)
    if skipped:
        print(f"[infer] {task}: skipped {len(skipped)} over-length samples")
    return predict_encoded(model, tokenizer, encoded, task, cfg)
