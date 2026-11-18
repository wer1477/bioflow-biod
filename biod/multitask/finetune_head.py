"""Head-only fine-tune for one decision when the interleaved run under-trains it.

Loads a multitask checkpoint with the LoRA adapter frozen, unfreezes only the
target head, and trains on that task for a few epochs. Useful for relay/role,
where a late or hard task can collapse to the majority class.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from transformers import AutoTokenizer

from biod.train.config import TrainConfig

from .catalog import load_catalog_snapshot
from .collate import Encoded, batch_key, encode_sample, pad_batch
from .data import load_split
from .infer import predict_samples
from .metrics import context_metrics, relay_metrics, role_metrics
from .model import MultiTaskBiodModel
from .tasks import TASK_SPECS, TASKS

HEAD = {
    "tool_select": "tool_head",
    "kb_select": "kb_head",
    "context_retention": "context_head",
    "role_select": "role_head",
    "relay_select": "relay_head",
}


def _log(path, rec):
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261102)
    args = ap.parse_args(argv)
    task = args.task
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "finetune_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    with open(os.path.join(args.checkpoint, "model_config.json"), encoding="utf-8") as fh:
        mc = json.load(fh)
    cfg = TrainConfig(base_model=mc["base_model"], head_hidden=int(mc.get("head_hidden", 1024)),
                      head_dim=int(mc.get("head_dim", 256)), lora=True, gradient_checkpointing=False)
    catalog = load_catalog_snapshot()
    tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = MultiTaskBiodModel(cfg, device="cuda", init_adapter=os.path.join(args.checkpoint, "adapter"))
    model.load_heads(os.path.join(args.checkpoint, "heads.pt"))
    for p in model.parameters():
        p.requires_grad_(False)
    target = getattr(model, HEAD[task])
    for p in target.parameters():
        p.requires_grad_(True)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[finetune] {task} trainable={sum(p.numel() for p in trainable)}", flush=True)

    train = load_split(args.data_dir, task, "train", catalog)
    val = load_split(args.data_dir, task, "val", catalog)
    rng = random.Random(args.seed)
    encoded: list[Encoded] = []
    for s in train:
        e = encode_sample(s, tok, TASK_SPECS[task].max_len)
        if e is not None:
            encoded.append(e)
    by_key = {}
    for e in encoded:
        by_key.setdefault(batch_key(e), []).append(e)
    batches = []
    for key in by_key:
        group = by_key[key]
        rng.shuffle(group)
        for i in range(0, len(group), args.batch_size):
            batches.append(group[i:i + args.batch_size])

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total = max(1, len(batches) * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / max(1, int(0.05 * total))) * max(0.1, 1.0 - step / total))
    step = 0
    best = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(batches)
        running = []
        for batch in batches:
            tensors = pad_batch(batch, tok.pad_token_id, device="cuda")
            out = model(tensors)
            loss, stats = model.compute_loss(tensors, out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            running.append(stats["loss"])
            if step % 10 == 0:
                _log(log_path, {"event": "step", "epoch": epoch, "step": step,
                                "loss": sum(running) / len(running), "lr": sched.get_last_lr()[0]})
                running = []
        preds = predict_samples(model, tok, val, task, TASK_SPECS[task].max_len, {"kb_tau": 0.5, "kb_top_k": 3})
        by_id = {p["id"]: p for p in preds}
        if task == "role_select":
            recs = [{"gold": (s.label or {})["role"], "pred": by_id[s.id]["role"],
                     "generator": s.meta.get("generator", "")} for s in val if s.id in by_id]
            metrics = role_metrics(recs); primary = metrics["accuracy_5"]
        elif task == "relay_select":
            recs = [{"gold": (s.label or {})["verdict"], "pred": by_id[s.id]["verdict"]} for s in val if s.id in by_id]
            metrics = relay_metrics(recs); primary = metrics["accuracy_3"]
        elif task == "context_retention":
            gold, pred = [], []
            for s in val:
                if s.id not in by_id:
                    continue
                pm = {x["seq"]: x["label"] for x in by_id[s.id]["labels"]}
                for x in s.label:
                    if x["seq"] in pm:
                        gold.append(x["label"]); pred.append(pm[x["seq"]])
            metrics = context_metrics([{"gold": g, "pred": p} for g, p in zip(gold, pred)]); primary = metrics["accuracy_3"]
        else:
            metrics = {}; primary = 0.0
        _log(log_path, {"event": "epoch", "epoch": epoch, "step": step, "primary": primary, "metrics": metrics})
        if primary > best:
            best = primary
            best_state = {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}
    if best_state is not None:
        target.load_state_dict(best_state)
    model.save_checkpoint(os.path.join(args.out_dir, "best"), meta={"task": task, "primary": best})
    with open(os.path.join(args.out_dir, "finetune_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": os.path.abspath(args.checkpoint), "task": task, "best_primary": best,
                   "config": vars(args)}, fh, ensure_ascii=False, indent=1)
    print("[finetune] done", best, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
