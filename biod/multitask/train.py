"""Multitask SFT for the five biod decisions.

Training is task-homogeneous per batch: a tool batch never mixes roles with
different candidate counts, and a context batch keeps one group-padding shape.
Optimizer steps cycle through tasks so every head sees the same number of
updates per epoch regardless of dataset size.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HOME", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), ".hf_cache"))
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Sequence

import torch
from transformers import AutoTokenizer

from biod.train.config import TrainConfig

from .catalog import load_catalog_snapshot
from .collate import Encoded, batch_key, encode_sample, make_order, pad_batch
from .data import dataset_manifest, load_split, sha256_file
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
from .model import MultiTaskBiodModel
from .tasks import TASK_SPECS, TASKS

# micro-batch / accumulation per task; effective batch ~16-32
BATCH_PLAN = {
    "tool_select": {"micro": 4, "accum": 2},
    "kb_select": {"micro": 2, "accum": 4},
    "context_retention": {"micro": 8, "accum": 1},
    "role_select": {"micro": 8, "accum": 1},
    "relay_select": {"micro": 8, "accum": 1},
    "premise_judge": {"micro": 8, "accum": 1},
    "overclaim_flag": {"micro": 8, "accum": 1},
}

PRIMARY = {
    "tool_select": "top1_strict",
    "kb_select": "hit@1",
    "context_retention": "accuracy_3",
    "role_select": "accuracy_5",
    "relay_select": "accuracy_3",
    "premise_judge": "macro_f1",
    "overclaim_flag": "micro_f1",
}


def _log(path: str, record: dict) -> None:
    record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def build_encoded(samples, tokenizer, task: str, max_len: int, shuffle: bool, rng: random.Random | None):
    out: list[Encoded] = []
    dropped = 0
    for sample in samples:
        order = None
        if shuffle and task in {"tool_select", "kb_select"}:
            order = make_order(len(sample.options), rng)
        enc = encode_sample(sample, tokenizer, max_len, order=order)
        if enc is None:
            dropped += 1
        else:
            out.append(enc)
    return out, dropped


def batches_for(encoded: Sequence[Encoded], task: str) -> list[list[Encoded]]:
    by_key = defaultdict(list)
    for enc in encoded:
        by_key[batch_key(enc)].append(enc)
    micro = BATCH_PLAN[task]["micro"]
    batches: list[list[Encoded]] = []
    for key in sorted(by_key, key=str):
        group = by_key[key]
        for i in range(0, len(group), micro):
            batches.append(group[i:i + micro])
    return batches


def validate(model, tokenizer, samples_by_task: dict[str, list], cfg: dict) -> dict:
    out = {}
    for task, samples in samples_by_task.items():
        if not samples:
            continue
        preds = predict_samples(model, tokenizer, samples, task, TASK_SPECS[task].max_len, cfg)
        by_id = {p["id"]: p for p in preds}
        if task == "tool_select":
            records = []
            for s in samples:
                p = by_id.get(s.id)
                if not p:
                    continue
                records.append({
                    "gold_id": (s.label or {}).get("id"),
                    "chosen_id": p["chosen_id"],
                    "ranked_ids": p["ranked_ids"],
                    "confidence": p["confidence"],
                    "no_tool_prob": p["no_tool_prob"],
                })
            out[task] = tool_metrics(records)
        elif task == "kb_select":
            records = []
            for s in samples:
                p = by_id.get(s.id)
                if not p:
                    continue
                grades = (s.label or {}).get("grades") or {}
                records.append({
                    "gold_ids": [k for k, v in grades.items() if v == 2],
                    "selected_ids": p["selected_ids"],
                    "scores": p["scores"],
                })
            out[task] = kb_metrics(records)
        elif task == "context_retention":
            gold, pred = [], []
            for s in samples:
                p = by_id.get(s.id)
                if not p:
                    continue
                pred_by_seq = {x["seq"]: x["label"] for x in p["labels"]}
                for x in s.label:
                    if x["seq"] in pred_by_seq:
                        gold.append(x["label"])
                        pred.append(pred_by_seq[x["seq"]])
            out[task] = context_metrics([{"gold": g, "pred": p} for g, p in zip(gold, pred)])
        elif task == "role_select":
            records = []
            for s in samples:
                p = by_id.get(s.id)
                if not p:
                    continue
                records.append({"gold": (s.label or {})["role"], "pred": p["role"], "generator": s.meta.get("generator", "")})
            out[task] = role_metrics(records)
        elif task == "relay_select":
            records = []
            for s in samples:
                p = by_id.get(s.id)
                if not p:
                    continue
                records.append({"gold": (s.label or {})["verdict"], "pred": p["verdict"]})
            out[task] = relay_metrics(records)
        elif task == "premise_judge":
            records = []
            for s in samples:
                p = by_id.get(s.id)
                if not p:
                    continue
                records.append({"gold": (s.label or {})["relation"], "pred": p["relation"], "confidence": p["confidence"]})
            out[task] = premise_metrics(records)
        elif task == "overclaim_flag":
            records = []
            for s in samples:
                p = by_id.get(s.id)
                if not p:
                    continue
                records.append({"gold_flags": (s.label or {}).get("flags") or [], "pred_flags": p["flags"]})
            out[task] = flag_metrics(records)
    return out


def score_of(metrics: dict) -> float:
    vals = []
    for task, key in PRIMARY.items():
        if task in metrics and metrics[task]:
            v = metrics[task].get(key)
            if v is not None:
                vals.append(float(v))
    return sum(vals) / len(vals) if vals else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--head-hidden", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--kb-tau", type=float, default=0.5)
    parser.add_argument("--kb-top-k", type=int, default=3)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--init-heads", default=None)
    parser.add_argument("--train-heads", default="")
    args = parser.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in TASK_SPECS:
            parser.error(f"unknown task {task!r}")
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    cfg = TrainConfig(
        base_model=args.base_model,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        head_hidden=args.head_hidden,
        dtype="bfloat16",
        seed=args.seed,
    )
    catalog = load_catalog_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = {task: {} for task in tasks}
    for task in tasks:
        for split in ("train", "val"):
            try:
                data[task][split] = load_split(args.data_dir, task, split, catalog, limit=args.val_limit if split == "val" else 0)
            except FileNotFoundError:
                data[task][split] = []
        print(f"[data] {task}: train={len(data[task]['train'])} val={len(data[task]['val'])}", flush=True)

    model = MultiTaskBiodModel(cfg, device=args.device, init_adapter=args.init_adapter)
    if args.init_heads:
        model.load_heads(args.init_heads)
    if args.train_heads:
        keep = {h.strip() for h in args.train_heads.split(",") if h.strip()}
        tops = {n.split(".")[0] for n, _ in model.named_parameters()}
        missing = keep - tops
        if missing:
            raise SystemExit(f"train-heads not found: {sorted(missing)}")
        for name, param in model.named_parameters():
            param.requires_grad_(name.split(".")[0] in keep)
    model.train()
    print("[model]", json.dumps(model.parameter_summary(), ensure_ascii=False), flush=True)

    optimizer = torch.optim.AdamW(model.optimizer_groups(), betas=(0.9, 0.999), eps=1e-8)
    if args.max_steps:
        total_est = args.max_steps
    else:
        total_est = 0
        for t in tasks:
            eff = BATCH_PLAN[t]["micro"] * BATCH_PLAN[t]["accum"]
            total_est += max(1, math.ceil(len(data[t]["train"]) / eff))
        total_est = max(1, total_est * args.epochs)
    warmup = max(1, int(0.03 * total_est))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * max(0.0, 1.0 - step / max(1, total_est))
    )

    rng = random.Random(args.seed)
    best_score = -1.0
    patience = 0
    global_step = 0
    stop = False
    for epoch in range(1, args.epochs + 1):
        groups_by_task: dict[str, list[list[list]]] = {}
        for task in tasks:
            samples = data[task]["train"]
            if not samples:
                continue
            encoded, dropped = build_encoded(samples, tokenizer, task, TASK_SPECS[task].max_len, not args.no_shuffle, rng)
            batches = batches_for(encoded, task)
            rng.shuffle(batches)
            accum = BATCH_PLAN[task]["accum"]
            groups = [batches[i:i + accum] for i in range(0, len(batches), accum)]
            groups_by_task[task] = groups
            print(f"[epoch {epoch}] {task}: samples={len(samples)} encoded={len(encoded)} dropped={dropped} "
                  f"batches={len(batches)} steps={len(groups)}", flush=True)

        order = [t for t in tasks if groups_by_task.get(t)]
        rng.shuffle(order)
        cursor = {t: 0 for t in order}
        active = list(order)
        while active and not stop:
            next_active = []
            for task in active:
                groups = groups_by_task[task]
                idx = cursor[task]
                if idx >= len(groups):
                    continue
                group = groups[idx]
                cursor[task] = idx + 1
                optimizer.zero_grad(set_to_none=True)
                running = []
                for batch in group:
                    tensors = pad_batch(batch, tokenizer.pad_token_id, device=args.device)
                    out = model(tensors)
                    loss, stats = model.compute_loss(tensors, out)
                    if not math.isfinite(stats["loss"]):
                        raise RuntimeError(f"non-finite loss at step {global_step} ({task})")
                    (loss / max(1, len(group))).backward()
                    running.append(stats["loss"])
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                _log(log_path, {"event": "step", "epoch": epoch, "task": task, "step": global_step,
                                "loss": sum(running) / max(1, len(running)),
                                "lr": scheduler.get_last_lr()[0]})
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break
                if cursor[task] < len(groups):
                    next_active.append(task)
            active = next_active

        metrics = validate(model, tokenizer, {t: data[t]["val"] for t in tasks},
                           {"kb_tau": args.kb_tau, "kb_top_k": args.kb_top_k})
        model.train()
        score = score_of(metrics)
        summary = {"event": "epoch", "epoch": epoch, "step": global_step, "score": score, "metrics": metrics}
        _log(log_path, summary)
        if score > best_score:
            best_score = score
            patience = 0
            model.save_checkpoint(os.path.join(args.out_dir, "best"), meta={"epoch": epoch, "score": score, "metrics": metrics})
        else:
            patience += 1
        if stop or patience >= 2:
            break
    model.save_checkpoint(os.path.join(args.out_dir, "last"), meta={"epoch": args.epochs, "score": best_score})

    manifest = {
        "schemaVersion": 1,
        "base_model": args.base_model,
        "catalogHash": catalog.sha256,
        "data": dataset_manifest(args.data_dir, tasks),
        "dataDir": os.path.abspath(args.data_dir),
        "splits": {t: {s: len(data[t].get(s, [])) for s in ("train", "val")} for t in tasks},
        "config": {
            "epochs": args.epochs, "max_steps": args.max_steps, "lr_lora": args.lr_lora,
            "lr_head": args.lr_head, "head_hidden": args.head_hidden, "seed": args.seed,
            "batch_plan": {t: BATCH_PLAN[t] for t in tasks}, "tasks": tasks,
            "max_len": {t: TASK_SPECS[t].max_len for t in tasks},
        },
        "best_score": best_score,
        "global_step": global_step,
        "parameter_summary": model.parameter_summary(),
    }
    with open(os.path.join(args.out_dir, "train_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("[done]", json.dumps({"best_score": best_score, "steps": global_step}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
