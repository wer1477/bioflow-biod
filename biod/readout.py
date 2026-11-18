"""Logit readouts: the zero-training analogue of the model's two heads.

``letter_choice``     listwise, one prompt, softmax over ``A..Z`` (+ NO_TOOL)
``pointwise_scores``  one prompt per candidate, absolute P(useful)

Both are *zero-training*: the base model's LM head is used directly, no LoRA,
no pointer head, no gradient steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .backend import Backend
from .catalog import Option
from .render import (
    DEFAULT_CHOICE_INSTRUCTION,
    DEFAULT_SCORE_INSTRUCTION,
    render_choice,
    render_pointwise,
)

YES_TOKENS = ("Yes", "yes", " Yes", " yes")
NO_TOKENS = ("No", "no", " No", " no")


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    values = list(logits.values())
    mx = max(values)
    exps = {k: math.exp(v - mx) for k, v in logits.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


@dataclass
class ChoiceResult:
    """Result of a listwise choice (tool library layers)."""

    prompt: str
    labels: list[str]
    ids: list[str]
    probs: dict[str, float]              # id -> probability
    no_tool_prob: float = 0.0
    order: list[str] = field(default_factory=list)

    @property
    def best_id(self) -> str | None:
        if not self.probs:
            return None
        return max(self.probs, key=self.probs.get)

    def selected_id(self, min_prob: float = 0.0, no_tool_margin: float = 0.0) -> str | None:
        """Return the argmax id unless NO_TOOL wins or the score is too low.

        ``no_tool_margin`` shifts the NO_TOOL decision boundary: NO_TOOL only
        wins when ``p(NO_TOOL) > p(best) + no_tool_margin``.  A positive margin
        therefore makes abstention harder (more selections), a negative margin
        makes it easier.  The default ``0.0`` is the original behaviour.
        """
        best = self.best_id
        best_prob = (self.probs or {}).get(best, 0.0) if best is not None else 0.0
        if self.no_tool_prob > best_prob + no_tool_margin:
            return None
        if best is None or best_prob <= min_prob:
            return None
        return best

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "labels": self.labels,
            "probs": self.probs,
            "no_tool_prob": self.no_tool_prob,
            "best_id": self.best_id,
        }


def letter_choice(
    backend: Backend,
    state: str,
    options: Sequence[Option],
    instruction: str = DEFAULT_CHOICE_INSTRUCTION,
    add_no_tool: bool = True,
) -> ChoiceResult:
    """Listwise readout over letter labels (requires <= 26 options)."""
    prompt, labels, ids = render_choice(state, options, instruction=instruction, add_no_tool=add_no_tool)
    rows = backend.score_tokens([prompt], labels)
    logits = rows[0] if rows else {}
    probs = _softmax(logits)

    per_id: dict[str, float] = {}
    no_tool = 0.0
    for label, oid in zip(labels, ids):
        p = probs.get(label, 0.0)
        if oid == "__no_tool__":
            no_tool = p
        else:
            per_id[oid] = p
    return ChoiceResult(prompt=prompt, labels=labels, ids=ids, probs=per_id, no_tool_prob=no_tool, order=ids)


def pointwise_scores(
    backend: Backend,
    state: str,
    options: Sequence[Option],
    instruction: str = DEFAULT_SCORE_INSTRUCTION,
) -> list[float]:
    """Absolute per-candidate P(useful) via a Yes/No readout.

    Uses two token variants for Yes and No to be robust to tokenizer quirks;
    only the first resolvable variant per side is scored.
    """
    prompts = [render_pointwise(state, opt, instruction=instruction) for opt in options]
    if not prompts:
        return []
    # score each prompt with all Yes/No variants at once, then aggregate per side
    tokens = list(YES_TOKENS) + list(NO_TOKENS)
    rows = backend.score_tokens(prompts, tokens)
    scores: list[float] = []
    for row in rows:
        yes_logit = _best_logit(row, YES_TOKENS)
        no_logit = _best_logit(row, NO_TOKENS)
        if yes_logit is None and no_logit is None:
            scores.append(0.0)
            continue
        if yes_logit is None:
            scores.append(0.0)
            continue
        if no_logit is None:
            scores.append(1.0)
            continue
        # binary softmax over the two sides
        mx = max(yes_logit, no_logit)
        ey, en = math.exp(yes_logit - mx), math.exp(no_logit - mx)
        scores.append(ey / (ey + en))
    return scores


def _best_logit(row: dict[str, float], candidates: Sequence[str]) -> float | None:
    values = [row[t] for t in candidates if t in row]
    return max(values) if values else None
