"""Losses for the two pointer heads.

tool: listwise cross-entropy over the 62 slots (61 candidates + NO_TOOL).
kb:   binary cross-entropy with soft targets ``grade / 2`` over the 32 entries.

Both losses are computed in float32 even when the body runs in bf16 autocast.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tool_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Listwise CE for the tool pointer head.

    Parameters
    ----------
    logits: ``[B, 62]`` masked-softmax inputs (all slots valid in v0, but a mask
        is accepted so slot subsets can be supported later).
    labels: ``[B]`` slot index of the gold option (``61`` = NO_TOOL).
    mask: optional ``[B, 62]`` bool tensor of valid slots.
    """
    logits = logits.float()
    labels = labels.long()
    if mask is None:
        return F.cross_entropy(logits, labels)
    masked = logits.masked_fill(~mask.bool(), float("-inf"))
    return F.cross_entropy(masked, labels)


def kb_bce_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE-with-logits over 32 entries with soft targets in ``[0, 1]``."""
    logits = logits.float()
    soft_targets = soft_targets.float()
    per_entry = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")
    if mask is None:
        return per_entry.mean()
    mask = mask.bool()
    denom = mask.sum().clamp(min=1).to(per_entry.dtype)
    return (per_entry * mask).sum() / denom


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax restricted to valid slots (used by the smoke normalisation gate)."""
    logits = logits.float()
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
    return torch.softmax(logits, dim=-1)
