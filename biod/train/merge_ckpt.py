"""Merge per-task heads / adapters from several checkpoints into one run dir.

Two-stage v0 training can leave the best tool head in one run and the best kb
head (or adapter) in another. This utility stitches them into a single
inference checkpoint without re-training::

    .venv/bin/python -m biod.train.merge_ckpt \
        --out artifacts/runs/biod-v0-final \
        --tool-ckpt artifacts/runs/biod-v0-lora \
        --kb-ckpt artifacts/runs/biod-v0-kb \
        --config-from artifacts/runs/biod-v0-lora \
        --adapter-from artifacts/runs/biod-v0-lora

It writes ``heads.pt``, ``biod_config.json``, and ``merge_manifest.json``. The
per-task query-position modes are taken from the checkpoint each head came from.
The adapter directory is copied when ``--adapter-from`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

import torch


def _resolve_heads(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "heads.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"heads.pt not found: {path}")
    return path


def _load_config(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "biod_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"biod_config.json not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _q_position(config: dict, task: str) -> str:
    value = config.get("q_position", "prompt")
    if isinstance(value, dict):
        return value.get(task, "prompt")
    return value


def merge(
    out_dir: str,
    tool_ckpt: str,
    kb_ckpt: str,
    config_from: str | None = None,
    adapter_from: str | None = None,
) -> dict:
    tool_payload = torch.load(_resolve_heads(tool_ckpt), map_location="cpu", weights_only=True)
    kb_payload = torch.load(_resolve_heads(kb_ckpt), map_location="cpu", weights_only=True)
    if int(tool_payload["hidden_size"]) != int(kb_payload["hidden_size"]):
        raise ValueError("tool and kb head hidden sizes differ")
    if int(tool_payload["head_dim"]) != int(kb_payload["head_dim"]):
        raise ValueError("tool and kb head dims differ")
    os.makedirs(out_dir, exist_ok=True)
    merged_heads = dict(tool_payload)
    merged_heads["tool"] = tool_payload["tool"]
    merged_heads["kb"] = kb_payload["kb"]
    torch.save(merged_heads, os.path.join(out_dir, "heads.pt"))

    tool_cfg = _load_config(tool_ckpt)
    kb_cfg = _load_config(kb_ckpt)
    base_cfg = _load_config(config_from) if config_from else {**tool_cfg}
    base_cfg = dict(base_cfg)
    base_cfg["q_position"] = {
        "tool": _q_position(tool_cfg, "tool"),
        "kb": _q_position(kb_cfg, "kb"),
    }
    base_cfg["merged_from"] = {
        "tool": os.path.abspath(tool_ckpt),
        "kb": os.path.abspath(kb_ckpt),
        "config": os.path.abspath(config_from) if config_from else os.path.abspath(tool_ckpt),
        "adapter": os.path.abspath(adapter_from) if adapter_from else None,
    }
    if adapter_from:
        lora = base_cfg.setdefault("lora", {})
        lora["enabled"] = True
    with open(os.path.join(out_dir, "biod_config.json"), "w", encoding="utf-8") as fh:
        json.dump(base_cfg, fh, ensure_ascii=False, indent=2)

    if adapter_from:
        src = os.path.join(adapter_from, "adapter") if os.path.isdir(os.path.join(adapter_from, "adapter")) else adapter_from
        dst = os.path.join(out_dir, "adapter")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    manifest = {
        "format": "biod-sft-v0-merge",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out": os.path.abspath(out_dir),
        "tool_ckpt": os.path.abspath(tool_ckpt),
        "kb_ckpt": os.path.abspath(kb_ckpt),
        "config_from": os.path.abspath(config_from) if config_from else None,
        "adapter_from": os.path.abspath(adapter_from) if adapter_from else None,
        "q_position": base_cfg["q_position"],
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    with open(os.path.join(out_dir, "merge_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tool-ckpt", required=True)
    p.add_argument("--kb-ckpt", required=True)
    p.add_argument("--config-from", default=None)
    p.add_argument("--adapter-from", default=None)
    args = p.parse_args(argv)
    manifest = merge(
        args.out,
        args.tool_ckpt,
        args.kb_ckpt,
        config_from=args.config_from,
        adapter_from=args.adapter_from,
    )
    print("MERGE_OK " + json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
