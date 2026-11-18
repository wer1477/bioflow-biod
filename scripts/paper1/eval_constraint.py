#!/usr/bin/env python3
"""E2 candidate-constraint evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer

from biod.multitask.model import MultiTaskBiodModel
from biod.multitask.paper1.constraint import evaluate_constraints, load_bases
from biod.train.config import TrainConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="Datasets/paper1/legacy-test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="open,authorized,postfilter,random_size")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    with open(os.path.join(args.checkpoint, "model_config.json"), encoding="utf-8") as fh:
        mc = json.load(fh)
    cfg = TrainConfig(base_model=mc["base_model"], head_hidden=int(mc.get("head_hidden", 1024)),
                      head_dim=int(mc.get("head_dim", 256)), lora=True, gradient_checkpointing=False)
    tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = MultiTaskBiodModel(cfg, device=args.device, init_adapter=os.path.join(args.checkpoint, "adapter"))
    model.load_heads(os.path.join(args.checkpoint, "heads.pt"))
    model.eval()
    bases = load_bases(args.data_dir, split=args.split)
    metrics, rows = evaluate_constraints(model, tok, bases, modes=[m.strip() for m in args.modes.split(",") if m.strip()],
                                         cfg={"kb_tau": 0.5, "kb_top_k": 3})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint), "split": args.split,
                   "data_dir": os.path.abspath(args.data_dir), "metrics": metrics}, fh, ensure_ascii=False, indent=1)
    pred_path = os.path.splitext(args.out)[0] + "_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"[write] {args.out} {pred_path} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
