#!/usr/bin/env python3
"""Evaluate a multitask checkpoint on a frozen split (7 heads)."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from biod.multitask.catalog import load_catalog_snapshot
from biod.multitask.data import load_split
from biod.multitask.evaluate import evaluate_student
from biod.multitask.model import MultiTaskBiodModel
from biod.multitask.tasks import TASKS
from biod.train.config import TrainConfig


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(checkpoint: str, catalog_hash: str) -> dict:
    fp = {"checkpoint": os.path.abspath(checkpoint), "catalogHash": catalog_hash}
    for rel in ("heads.pt", "model_config.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
        path = os.path.join(checkpoint, rel)
        if os.path.isfile(path):
            fp[rel.replace("/", "_") + "_sha256"] = sha256_file(path)
    try:
        fp["git_commit"] = subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - fingerprint is best-effort
        fp["git_commit"] = ""
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--predictions", default="")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    with open(os.path.join(args.checkpoint, "model_config.json"), encoding="utf-8") as fh:
        model_cfg = json.load(fh)
    cfg = TrainConfig(base_model=model_cfg["base_model"], head_hidden=int(model_cfg.get("head_hidden", 1024)),
                      head_dim=int(model_cfg.get("head_dim", 256)), lora=True, gradient_checkpointing=False)
    catalog = load_catalog_snapshot()
    tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = MultiTaskBiodModel(cfg, device=args.device, init_adapter=os.path.join(args.checkpoint, "adapter"))
    model.load_heads(os.path.join(args.checkpoint, "heads.pt"))
    model.eval()
    data = {}
    for task in tasks:
        data[task] = load_split(args.data_dir, task, args.split, catalog)
        print(f"[eval] {task}: {len(data[task])} samples", flush=True)

    preds: list[dict] = []
    metrics = evaluate_student(model, tok, data, {"kb_tau": 0.5, "kb_top_k": 3}, predictions_out=preds)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "fingerprint": fingerprint(args.checkpoint, catalog.sha256),
        "metrics": metrics,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    pred_path = args.predictions or os.path.splitext(args.out)[0] + "_predictions.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(pred_path)), exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as fh:
        for row in preds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"[write] {args.out} {pred_path} rows={len(preds)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
