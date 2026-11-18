"""Biod SFT v0 training package.

Trains two pointer heads (tool choice over 61 tools + NO_TOOL; knowledge-base
scoring over 32 entries) on top of a frozen ``Qwen/Qwen3.5-4B-Base`` text tower,
optionally with a rank-16 LoRA adapter.

Entry points
------------
``biod.train.config.TrainConfig``   hyper-parameters (contract §3 defaults)
``biod.train.data``                 JSONL loaders + catalog validation
``biod.train.collate``              rendering, option shuffle, token spans, padding
``biod.train.loss``                 listwise CE (tool) + soft-target BCE (kb)
``biod.train.model``                frozen text tower + LoRA + two pointer heads
``biod.train.infer.BiodPredictor``  inference API used by the eval harness
"""

from __future__ import annotations

__all__ = [
    "TrainConfig",
    "BiodModel",
    "BiodPredictor",
    "PointerPrompt",
]

RENDER_VERSION = 1
CKPT_FORMAT = "biod-sft-v0"
