"""Shared evaluation: student predictions vs Qwen baseline, same records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from .baseline_qwen import predict_sample as qwen_predict
from .data import Sample
from .infer import predict_samples
from .metrics import (
    context_metrics,
    flag_metrics,
    kb_metrics,
    premise_metrics,
    relay_metrics,
    role_metrics,
    tool_metrics,
)
from .tasks import TASK_SPECS


def records_for(task: str, samples: Sequence[Sample], preds: Sequence[dict]) -> list[dict]:
    by_id = {p["id"]: p for p in preds}
    out: list[dict] = []
    for s in samples:
        p = by_id.get(s.id)
        if not p:
            continue
        if task == "tool_select":
            out.append({
                "gold_id": (s.label or {}).get("id"),
                "chosen_id": p.get("chosen_id"),
                "ranked_ids": p.get("ranked_ids", []),
                "confidence": p.get("confidence", 0.0),
            })
        elif task == "kb_select":
            grades = (s.label or {}).get("grades") or {}
            out.append({
                "gold_ids": [k for k, v in grades.items() if v == 2],
                "selected_ids": p.get("selected_ids", []),
                "scores": p.get("scores", {}),
            })
        elif task == "context_retention":
            pred_by_seq = {x["seq"]: x["label"] for x in p.get("labels", [])}
            for x in s.label:
                if x["seq"] in pred_by_seq:
                    out.append({"gold": x["label"], "pred": pred_by_seq[x["seq"]]})
        elif task == "role_select":
            out.append({"gold": (s.label or {})["role"], "pred": p.get("role"), "generator": s.meta.get("generator", "")})
        elif task == "relay_select":
            out.append({"gold": (s.label or {})["verdict"], "pred": p.get("verdict")})
        elif task == "premise_judge":
            out.append({
                "gold": (s.label or {})["relation"],
                "pred": p.get("relation"),
                "confidence": p.get("confidence", 0.0),
            })
        elif task == "overclaim_flag":
            out.append({
                "gold_flags": (s.label or {}).get("flags") or [],
                "pred_flags": p.get("flags", []),
            })
    return out


def compute(task: str, samples: Sequence[Sample], preds: Sequence[dict]) -> dict:
    recs = records_for(task, samples, preds)
    if task == "tool_select":
        return tool_metrics(recs)
    if task == "kb_select":
        return kb_metrics(recs)
    if task == "context_retention":
        return context_metrics(recs)
    if task == "role_select":
        return role_metrics(recs)
    if task == "relay_select":
        return relay_metrics(recs)
    if task == "premise_judge":
        return premise_metrics(recs)
    if task == "overclaim_flag":
        return flag_metrics(recs)
    raise ValueError(task)


def evaluate_student(
    model,
    tokenizer,
    samples_by_task: dict,
    cfg: dict | None = None,
    predictions_out: list | None = None,
) -> dict:
    """Evaluate a student checkpoint.

    ``predictions_out`` is optional; when given, per-item predictions are
    appended as ``{"task": <task>, ...}`` so the caller can persist a JSONL
    artifact for paired/bootstrap analysis.
    """
    out = {}
    for task, samples in samples_by_task.items():
        if not samples:
            continue
        preds = predict_samples(model, tokenizer, samples, task, TASK_SPECS[task].max_len, cfg)
        if predictions_out is not None:
            for pred in preds:
                predictions_out.append({"task": task, **pred})
        out[task] = compute(task, samples, preds)
    return out


def evaluate_qwen(samples_by_task: dict, workers: int = 4) -> dict:
    out = {}
    for task, samples in samples_by_task.items():
        if not samples:
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            preds = list(pool.map(qwen_predict, samples))
        for pred, sample in zip(preds, samples):
            pred["id"] = sample.id
        out[task] = compute(task, samples, preds)
        lat = sorted(float(p.get("latency_ms", 0.0)) for p in preds)
        if lat:
            out[task]["p50_ms"] = lat[len(lat) // 2]
    return out
