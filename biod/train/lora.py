"""Shared LoRA application for the v0 two-head model and the multitask model.

Extracted from ``BiodModel._apply_lora`` so both models validate peft target
matching identically. peft silently ignores target names that match nothing;
silent misses would leave part of the tower unadapted and are the kind of bug
that only shows up as a slightly worse metric, so every target must match the
same module count as the frozen base tower.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn


def suffix_counts(module: nn.Module, targets: Iterable[str]) -> dict[str, int]:
    counts = {name: 0 for name in targets}
    for name, _ in module.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in counts:
            counts[suffix] += 1
    return counts


def apply_lora(text: nn.Module, cfg: Any, base_counts: dict[str, int]) -> tuple[nn.Module, dict[str, int]]:
    """Wrap ``text`` with LoRA; return ``(wrapped, matched_counts)``."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - environment error
        raise ImportError(
            "peft is required for LoRA training; install peft==0.21.2 "
            "(or run with --head-only)"
        ) from exc

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.lora_targets),
        task_type="FEATURE_EXTRACTION",
    )
    wrapped = get_peft_model(text, lora_cfg)
    matched = {name: 0 for name in cfg.lora_targets}
    for name, _ in wrapped.named_modules():
        for target in cfg.lora_targets:
            if name.endswith(f".{target}.base_layer"):
                matched[target] += 1
    missing = [t for t, n in matched.items() if n == 0]
    mismatch = {
        t: (matched[t], base_counts[t])
        for t in cfg.lora_targets
        if matched[t] != base_counts[t]
    }
    if missing or mismatch:
        raise RuntimeError(
            "LoRA target validation failed: "
            f"missing={missing}, count_mismatch={mismatch}, "
            f"base_counts={base_counts}"
        )
    return wrapped, matched
