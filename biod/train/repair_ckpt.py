"""Repair a checkpoint whose ``biod_config.json`` / ``train_manifest.json`` were
not written (e.g. a crash after the weights were saved).

The weight files (``heads.pt``, ``adapter/``) and ``train_log.jsonl`` are the
sources of truth; this utility rebuilds the two config/manifest JSON files from
them plus the frozen catalog, so a crashed run does not need to be retrained.

    .venv/bin/python -m biod.train.repair_ckpt --run-dir artifacts/runs/biod-v0-lora \
        --q-tool state --q-kb prompt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import torch

from biod.catalog import load_knowledge_base, load_tool_library

from .config import (
    KB_CATALOG_DIR,
    KB_CATALOG_VERSION,
    NO_TOOL_ID,
    TOOL_CATALOG_DIR,
    TOOL_CATALOG_VERSION,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_log(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "train_log.jsonl")
    events = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def _catalog_options(task: str) -> list[dict]:
    if task == "tool":
        domains = load_tool_library(os.path.join(_REPO_ROOT, TOOL_CATALOG_DIR))
    else:
        domains = load_knowledge_base(os.path.join(_REPO_ROOT, KB_CATALOG_DIR))
    return [
        {"id": o.id, "name": o.name, "description": o.description, "domain": o.domain}
        for d in domains
        for o in d.options
    ]


def repair(run_dir: str, q_tool: str = "prompt", q_kb: str = "prompt") -> dict:
    events = _read_log(run_dir)
    run_start = next((e for e in events if e.get("event") == "run_start"), {})
    cfg = run_start.get("config", {})
    task = cfg.get("task", "both")
    tasks = ["tool", "kb"] if task == "both" else [task]

    heads_path = os.path.join(run_dir, "heads.pt")
    if not os.path.isfile(heads_path):
        raise FileNotFoundError(f"missing {heads_path} (nothing to repair)")
    heads = torch.load(heads_path, map_location="cpu", weights_only=True)
    adapter_dir = os.path.join(run_dir, "adapter")
    lora_enabled = os.path.isdir(adapter_dir)

    tool_options = _catalog_options("tool")
    kb_options = _catalog_options("kb")
    config = {
        "format": "biod-sft-v0",
        "biod_config_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": cfg.get("run_id", os.path.basename(run_dir)),
        "repaired": True,
        "base_model": cfg.get("base_model") or "Qwen/Qwen3.5-4B-Base",
        "base_model_path": cfg.get("base_model_path")
        or cfg.get("base_model")
        or "Qwen/Qwen3.5-4B-Base",
        "text_submodule": "language_model",
        "dtype": cfg.get("dtype", "bfloat16"),
        "hidden_size": int(heads["hidden_size"]),
        "head_dim": int(heads["head_dim"]),
        "head_dropout": cfg.get("head_dropout", 0.05),
        "head_norm_eps": cfg.get("head_norm_eps", 1e-5),
        "head_weight_decay": cfg.get("head_weight_decay", 0.01),
        "q_position": {"tool": q_tool if "tool" in tasks else "prompt",
                        "kb": q_kb if "kb" in tasks else "prompt"},
        "q_position_run": cfg.get("q_position", "prompt"),
        "ready_for_inference": True,
        "max_len": int(cfg.get("max_len") or 4096),
        "seed": cfg.get("seed", 20261022),
        "render_version": 1,
        "git": run_start.get("git", {}),
        "catalog": {
            "tool": {"version": TOOL_CATALOG_VERSION, "num_options": len(tool_options),
                     "no_tool_id": NO_TOOL_ID},
            "kb": {"version": KB_CATALOG_VERSION, "num_options": len(kb_options)},
        },
        "tool": {"num_options": len(tool_options), "no_tool_id": NO_TOOL_ID,
                 "option_ids": [o["id"] for o in tool_options], "options": tool_options},
        "kb": {"num_options": len(kb_options),
               "option_ids": [o["id"] for o in kb_options], "options": kb_options},
        "lora": {
            "enabled": lora_enabled,
            "r": cfg.get("lora_r", 16),
            "alpha": cfg.get("lora_alpha", 32),
            "dropout": cfg.get("lora_dropout", 0.05),
            "targets": cfg.get("lora_targets"),
        },
    }
    with open(os.path.join(run_dir, "biod_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)

    step_events = [e for e in events if e.get("event") == "step"]
    val_events = [e for e in events if e.get("event") in {"val_epoch", "val"}]
    train_done = next((e for e in events if e.get("event") == "train_done"), {})
    manifest = {
        "run_id": config["run_id"],
        "format": "biod-sft-v0",
        "repaired": True,
        "repaired_at": config["created_at"],
        "head_only": not lora_enabled,
        "git": run_start.get("git", {}),
        "gpu": run_start.get("gpu", {}),
        "host": platform.node(),
        "catalog_version": {"tool": TOOL_CATALOG_VERSION, "kb": KB_CATALOG_VERSION},
        "hyperparams": cfg,
        "data": {},
        "loss": {
            "history": [e.get("loss") for e in step_events],
            "first_avg": train_done.get("loss_first"),
            "last_avg": train_done.get("loss_last"),
        },
        "val": (val_events[-1] if val_events else {}),
        "timing": {
            "elapsed_s": train_done.get("elapsed_s"),
            "optimizer_steps": train_done.get("steps"),
            "tokens_seen": train_done.get("tokens_seen"),
            "tokens_per_s": train_done.get("tokens_per_s"),
            "max_mem_gb": train_done.get("max_mem_gb"),
        },
        "notes": "repaired from train_log.jsonl after a checkpoint-save crash",
        "status": "repaired",
    }
    with open(os.path.join(run_dir, "train_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return {"run_dir": os.path.abspath(run_dir), "lora": lora_enabled,
            "q_position": config["q_position"], "max_len": config["max_len"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--q-tool", choices=["prompt", "state"], default="prompt")
    p.add_argument("--q-kb", choices=["prompt", "state"], default="prompt")
    args = p.parse_args(argv)
    info = repair(args.run_dir, q_tool=args.q_tool, q_kb=args.q_kb)
    print("REPAIR_OK " + json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
