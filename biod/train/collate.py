"""Examples, option shuffling and batching for the pointer heads.

Pipeline per record (contract §3):

1. render the pointer prompt (``biod.render.render_tool_prompt`` /
   ``render_kb_prompt``) with the option order for this epoch;
2. tokenize with ``add_special_tokens=False`` and ``return_offsets_mapping``;
3. map each option line's character span to the index of its **last token**
   (``biod.render.char_spans_to_token_indices``);
4. remap the label to the shuffled display slot and assert that the slot's id
   is still the gold id (catches every shuffle/remap bug at data time);
5. if the prompt is longer than ``max_len`` → **drop** the example (never
   truncate; truncating silently drops option rows).

Tokenization is intentionally lazy on torch: pure-python parts of this module
import nothing from torch, so they can be unit-tested locally. A fake tokenizer
whose tokens are characters with identity offsets exercises the whole span
pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from biod.catalog import Option
from biod.render import (
    NO_TOOL_ID,
    PointerPrompt,
    canonical_order,
    char_end_to_token_index,
    char_spans_to_token_indices,
    render_kb_prompt,
    render_tool_prompt,
    reversed_order,
)

__all__ = [
    "PointerExample",
    "make_order",
    "encode_prompt",
    "build_tool_example",
    "build_kb_example",
    "build_examples",
    "pad_batch",
    "iter_batches",
    "token_lengths",
    "expected_option_count",
]


def expected_option_count(task: str, num_candidates: int) -> int:
    """Pointer slots: tool = candidates + NO_TOOL row; kb = candidates."""
    return num_candidates + (1 if task == "tool" else 0)


def make_order(
    n: int,
    rng: random.Random | None = None,
    order: Sequence[int] | None = None,
) -> list[int]:
    """Return a display order (list of canonical indices)."""
    if order is not None:
        order = list(order)
        if sorted(order) != list(range(n)):
            raise ValueError(f"order must be a permutation of 0..{n - 1}")
        return order
    if rng is None:
        return canonical_order(n)
    perm = canonical_order(n)
    rng.shuffle(perm)
    return perm


def order_from_name(name: str, n: int) -> list[int] | None:
    """``canonical`` / ``reversed`` / ``None`` (random) helper for eval paths."""
    if name is None or name == "canonical":
        return canonical_order(n)
    if name == "reversed":
        return reversed_order(n)
    if name == "random":
        return None
    raise ValueError(f"unknown order {name!r}; expected canonical|reversed|random")


@dataclass
class PointerExample:
    """One fully prepared training/eval example (unpadded)."""

    task: str
    record_id: str
    input_ids: list[int]
    option_positions: list[int]  # token index of each option line's last token
    q_position: int  # token index of the query (last token of the prompt)
    prompt_len: int
    option_ids: list[str]  # display order (tool: NO_TOOL last)
    canonical_indices: list[int]  # display slot -> canonical index (-1 for NO_TOOL)
    order: list[int] = field(default_factory=list)  # canonical indices in display order
    tool_label: int | None = None  # display slot of the gold tool (61 = NO_TOOL)
    kb_soft: list[float] | None = None  # display-slot soft labels in [0, 1]
    gold_id: str | None = None
    gold_acceptable: list[str] = field(default_factory=list)
    num_positive: int = 0

    @property
    def num_slots(self) -> int:
        return len(self.option_positions)

    def label_check(self) -> bool:
        """True when the remapped label still points at the gold id."""
        if self.task != "tool" or self.tool_label is None:
            return True
        return self.option_ids[self.tool_label] == self.gold_id

    def to_debug_dict(self) -> dict:
        return {
            "task": self.task,
            "record_id": self.record_id,
            "prompt_len": self.prompt_len,
            "num_slots": self.num_slots,
            "order": list(self.order),
            "gold_id": self.gold_id,
            "tool_label": self.tool_label,
            "label_ok": self.label_check(),
        }


def encode_prompt(
    prompt: PointerPrompt,
    tokenizer: Any,
    max_len: int = 0,
) -> tuple[list[int], list[int], int] | None:
    """Tokenize a pointer prompt and resolve option-line token positions.

    Returns ``(input_ids, option_token_positions, state_last_token_index)`` or
    ``None`` when the prompt exceeds ``max_len`` (over-length samples are
    dropped, never truncated).
    """
    enc = tokenizer(
        prompt.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = list(enc["input_ids"])
    if max_len and len(ids) > max_len:
        return None
    offsets = enc["offset_mapping"]
    if len(offsets) != len(ids):
        raise ValueError("tokenizer returned inconsistent offset_mapping length")
    positions = char_spans_to_token_indices(offsets, prompt.option_spans)
    if len(positions) != prompt.num_options:
        raise ValueError(
            f"resolved {len(positions)} option positions for {prompt.num_options} spans"
        )
    if len(set(positions)) != len(positions):
        raise ValueError("two option lines resolved to the same token index")
    if prompt.state_char_span is None:
        raise ValueError("pointer prompt has no state_char_span")
    state_token = char_end_to_token_index(offsets, prompt.state_char_span[1])
    return ids, positions, state_token


def build_tool_example(
    record: dict,
    options: Sequence[Option],
    id_to_index: dict[str, int],
    tokenizer: Any,
    max_len: int = 0,
    order: Sequence[int] | None = None,
    q_position_mode: str = "prompt",
) -> PointerExample | None:
    """Prepare one tool-choice example; ``None`` if it exceeds ``max_len``."""
    prompt = render_tool_prompt(record["state_text"], options, order=order)
    encoded = encode_prompt(prompt, tokenizer, max_len=max_len)
    if encoded is None:
        return None
    ids, positions, state_token = encoded
    gold = record["gold_primary"]
    if gold == NO_TOOL_ID:
        label = prompt.meta["no_tool_slot"]
    else:
        canonical_idx = id_to_index[gold]
        label = prompt.canonical_indices.index(canonical_idx)
    if prompt.option_ids[label] != gold:
        raise AssertionError(
            f"{record['id']}: shuffled label mismatch ({prompt.option_ids[label]!r} != {gold!r})"
        )
    return PointerExample(
        task="tool",
        record_id=record["id"],
        input_ids=ids,
        option_positions=positions,
        q_position=state_token if q_position_mode == "state" else len(ids) - 1,
        prompt_len=len(ids),
        option_ids=list(prompt.option_ids),
        canonical_indices=list(prompt.canonical_indices),
        order=list(prompt.canonical_indices[:-1]),
        tool_label=label,
        gold_id=gold,
        gold_acceptable=list(record.get("gold_acceptable") or []),
    )


def build_kb_example(
    record: dict,
    options: Sequence[Option],
    id_to_index: dict[str, int],
    tokenizer: Any,
    max_len: int = 0,
    order: Sequence[int] | None = None,
    q_position_mode: str = "prompt",
) -> PointerExample | None:
    """Prepare one knowledge-base example; ``None`` if it exceeds ``max_len``."""
    prompt = render_kb_prompt(record["state_text"], options, order=order)
    encoded = encode_prompt(prompt, tokenizer, max_len=max_len)
    if encoded is None:
        return None
    ids, positions, state_token = encoded
    soft = [0.0] * len(options)
    for entry_id, value in record["soft_labels"].items():
        display_slot = prompt.canonical_indices.index(id_to_index[entry_id])
        soft[display_slot] = float(value)
    return PointerExample(
        task="kb",
        record_id=record["id"],
        input_ids=ids,
        option_positions=positions,
        q_position=state_token if q_position_mode == "state" else len(ids) - 1,
        prompt_len=len(ids),
        option_ids=list(prompt.option_ids),
        canonical_indices=list(prompt.canonical_indices),
        order=list(prompt.canonical_indices),
        kb_soft=soft,
        num_positive=int(record.get("num_positive", 0)),
    )


def build_examples(
    records: Sequence[dict],
    task: str,
    options: Sequence[Option],
    id_to_index: dict[str, int],
    tokenizer: Any,
    max_len: int = 0,
    rng: random.Random | None = None,
    shuffle_options: bool = False,
    q_position_mode: str = "prompt",
) -> tuple[list[PointerExample], dict]:
    """Build examples for one epoch, dropping over-length rows and counting them.

    Returns ``(examples, drop_stats)``. ``shuffle_options`` draws one option
    permutation per example from ``rng`` (reproducible when ``rng`` is seeded).
    """
    builder = build_tool_example if task == "tool" else build_kb_example
    n = len(options)
    examples: list[PointerExample] = []
    dropped: list[dict] = []
    for record in records:
        order = make_order(n, rng) if shuffle_options else None
        ex = builder(
            record, options, id_to_index, tokenizer,
            max_len=max_len, order=order, q_position_mode=q_position_mode,
        )
        if ex is None:
            dropped.append({"id": record["id"], "reason": "over_max_len"})
            continue
        examples.append(ex)
    stats = {
        "task": task,
        "built": len(examples),
        "dropped": len(dropped),
        "dropped_ids": [d["id"] for d in dropped[:20]],
        "dropped_total": len(dropped),
    }
    return examples, stats


def token_lengths(
    records: Sequence[dict],
    task: str,
    options: Sequence[Option],
    tokenizer: Any,
) -> list[int]:
    """Rendered token length per record (canonical order), for max_len stats."""
    lengths: list[int] = []
    for record in records:
        if task == "tool":
            prompt = render_tool_prompt(record["state_text"], options)
        else:
            prompt = render_kb_prompt(record["state_text"], options)
        enc = tokenizer(prompt.text, add_special_tokens=False, return_offsets_mapping=False, truncation=False)
        lengths.append(len(enc["input_ids"]))
    return lengths


# --------------------------------------------------------------------------- #
# batching (needs torch only here)
# --------------------------------------------------------------------------- #
def pad_batch(
    examples: Sequence[PointerExample],
    pad_token_id: int,
    device: str | None = None,
) -> dict:
    """Left-pad a list of examples into tensors for one forward pass.

    Left padding keeps the last position a real token for every row, so the
    query is always ``hidden[:, -1]``. Option positions are shifted by the pad
    width.
    """
    import torch

    if not examples:
        raise ValueError("empty batch")
    tasks = {ex.task for ex in examples}
    if len(tasks) != 1:
        raise ValueError(f"cannot mix tasks in one batch: {tasks}")
    task = examples[0].task
    batch_size = len(examples)
    max_len = max(ex.prompt_len for ex in examples)
    num_slots = examples[0].num_slots
    if any(ex.num_slots != num_slots for ex in examples):
        raise ValueError("inconsistent number of pointer slots in batch")

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    option_positions = torch.zeros((batch_size, num_slots), dtype=torch.long)
    for row, ex in enumerate(examples):
        if len(ex.input_ids) != ex.prompt_len:
            raise ValueError(f"{ex.record_id}: prompt_len does not match input_ids")
        pad = max_len - ex.prompt_len
        input_ids[row, pad:] = torch.tensor(ex.input_ids, dtype=torch.long)
        attention_mask[row, pad:] = 1
        option_positions[row] = torch.tensor(ex.option_positions, dtype=torch.long) + pad
        if not 0 <= ex.q_position < ex.prompt_len:
            raise ValueError(f"{ex.record_id}: q_position {ex.q_position} out of range")

    q_positions = torch.tensor(
        [ex.q_position + (max_len - ex.prompt_len) for ex in examples], dtype=torch.long
    )
    batch: dict = {
        "task": task,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "option_positions": option_positions,
        "q_position": q_positions,
        "record_ids": [ex.record_id for ex in examples],
        "prompt_lens": torch.tensor([ex.prompt_len for ex in examples], dtype=torch.long),
        "option_ids": [ex.option_ids for ex in examples],
        "canonical_indices": [ex.canonical_indices for ex in examples],
    }
    if task == "tool":
        batch["tool_labels"] = torch.tensor([ex.tool_label for ex in examples], dtype=torch.long)
        batch["gold_ids"] = [ex.gold_id for ex in examples]
        batch["gold_acceptable"] = [ex.gold_acceptable for ex in examples]
    else:
        import torch as _torch

        batch["kb_soft"] = _torch.tensor([ex.kb_soft for ex in examples], dtype=_torch.float32)
        batch["num_positive"] = [ex.num_positive for ex in examples]
    if device:
        for key, value in batch.items():
            if hasattr(value, "to"):
                batch[key] = value.to(device, non_blocking=True)
    return batch


def iter_batches(examples: Sequence[PointerExample], batch_size: int):
    """Yield consecutive mini-batches of examples."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])
