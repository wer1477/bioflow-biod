#!/usr/bin/env python
"""Evaluate role-decision heads against a labelled gold set.

Two modes:
  --student-ckpt DIR   score with the local checkpoint (uses the sidecar Predictor)
  --student-preds FILE reuse an existing prediction file

Writes per-item predictions (JSONL) and prints metrics. When both a student
prediction file and a baseline prediction file are given, prints the comparison
used for the release gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from biod.multitask.metrics import flag_metrics, premise_metrics  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def score_student(ckpt: str, task: str, items: list[dict], batch: int = 32) -> list[dict]:
    from biod.multitask.serve import Predictor

    predictor = Predictor(ckpt, "cuda")
    preds: list[dict] = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        payload = []
        for rec in chunk:
            entry = {"id": rec["id"]}
            if task == "premise_judge":
                entry.update({"left": rec["left"], "right": rec["right"],
                              "context": rec.get("context", ""), "unit": rec.get("unit", "claim")})
            else:
                entry.update({"left": rec["claim"], "right": rec["evidence"],
                              "context": rec.get("context", "")})
            payload.append(entry)
        result = predictor.pairs(task, payload)
        preds.extend(result["items"])
    return preds


def metrics_for(task: str, gold: list[dict], preds: list[dict]) -> dict:
    by_id = {p["id"]: p for p in preds}
    if task == "premise_judge":
        records = []
        for rec in gold:
            p = by_id.get(rec["id"])
            if not p:
                continue
            records.append({"gold": rec["label"]["relation"], "pred": p["label"],
                            "confidence": float(p["confidence"])})
        return premise_metrics(records)
    records = []
    for rec in gold:
        p = by_id.get(rec["id"])
        if not p:
            continue
        records.append({"gold_flags": rec["label"]["flags"], "pred_flags": p["flags"]})
    return flag_metrics(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["premise_judge", "overclaim_flag"])
    ap.add_argument("--gold", required=True)
    ap.add_argument("--student-ckpt", default=None)
    ap.add_argument("--student-preds", default=None)
    ap.add_argument("--baseline-preds", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    if args.student_preds:
        preds = load_jsonl(args.student_preds)
    else:
        if not args.student_ckpt:
            raise SystemExit("need --student-ckpt or --student-preds")
        preds = score_student(args.student_ckpt, args.task, gold)
        with open(args.out, "w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    student = metrics_for(args.task, gold, preds)
    print(json.dumps({"student": student}, ensure_ascii=False, indent=2))

    if args.baseline_preds:
        base = metrics_for(args.task, gold, load_jsonl(args.baseline_preds))
        print(json.dumps({"baseline_qwen38": base}, ensure_ascii=False, indent=2))
        key = "macro_f1" if args.task == "premise_judge" else "micro_f1"
        gap = float(student.get(key, 0)) - float(base.get(key, 0))
        print(f"\nstudent {key}={student.get(key):.4f}  baseline {key}={base.get(key):.4f}  gap={gap:+.4f}")
        print("GATE (student >= baseline - 5pt):", "PASS" if gap >= -0.05 else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
