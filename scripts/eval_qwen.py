#!/usr/bin/env python3
"""Evaluate Qwen3.8-Flash-Next on the same frozen test split."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biod.multitask.catalog import load_catalog_snapshot
from biod.multitask.data import load_split
from biod.multitask.evaluate import evaluate_qwen
from biod.multitask.tasks import TASKS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    catalog = load_catalog_snapshot()
    data = {}
    for task in tasks:
        data[task] = load_split(args.data_dir, task, args.split, catalog)
        print(f"[qwen] {task}: {len(data[task])} samples", flush=True)
    metrics = evaluate_qwen(data, workers=args.workers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"model": "Qwen3.8-Flash-Next-NVFP4", "split": args.split, "metrics": metrics}, fh, ensure_ascii=False, indent=1)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
