"""Encoding + batching for the five heads.

Pointer tasks (tool/kb) keep one token position per candidate line; the
group-3 head keeps one token position per cold-zone group. Everything is
left-padded so the query position is always the last real token.

Batches are homogeneous by ``(task, role, num_slots)`` for pointer tasks: the
candidate count is role-dependent (65..108 for tools), and padding a pointer
head over mixed candidate sets would silently change the softmax denominator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from biod.render import char_end_to_token_index, char_spans_to_token_indices

from .data import NO_TOOL, Sample
from .render import (
    render_context,
    render_kb,
    render_overclaim,
    render_premise,
    render_relay,
    render_role,
    render_tool,
)
from .tasks import CONTEXT_LABELS, OVERCLAIM_FLAGS, PREMISE_LABELS, RELAY_VERDICTS, ROLES

CONTEXT_INDEX = {label: i for i, label in enumerate(CONTEXT_LABELS)}
ROLE_INDEX = {role: i for i, role in enumerate(ROLES)}
VERDICT_INDEX = {v: i for i, v in enumerate(RELAY_VERDICTS)}
PREMISE_INDEX = {label: i for i, label in enumerate(PREMISE_LABELS)}


@dataclass
class Encoded:
    task: str
    record_id: str
    role: str | None
    input_ids: list[int]
    prompt_len: int
    q_position: int
    option_positions: list[int] = field(default_factory=list)
    option_ids: list[str] = field(default_factory=list)
    # pointer labels
    tool_label: int | None = None
    kb_soft: list[float] | None = None
    # group-3
    group_positions: list[int] = field(default_factory=list)
    group_labels: list[int] = field(default_factory=list)
    # classification
    class_label: int | None = None
    reason_label: int | None = None
    flag_label: list[float] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_slots(self) -> int:
        return len(self.option_positions)

    @property
    def num_groups(self) -> int:
        return len(self.group_positions)


def make_order(n: int, rng: random.Random | None = None) -> list[int]:
    order = list(range(n))
    if rng is not None:
        rng.shuffle(order)
    return order


def _tokenize(tokenizer: Any, text: str, max_len: int) -> tuple[list[int], list[tuple[int, int]]] | None:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = list(enc["input_ids"])
    if max_len and len(ids) > max_len:
        return None
    offsets = [tuple(int(x) for x in off) for off in enc["offset_mapping"]]
    if len(offsets) != len(ids):
        raise ValueError("tokenizer returned inconsistent offset_mapping length")
    return ids, offsets


def encode_pointer(
    sample: Sample,
    tokenizer: Any,
    max_len: int,
    order: Sequence[int] | None = None,
) -> Encoded | None:
    if sample.task == "tool_select":
        prompt = render_tool(sample.state_text, sample.options, role=sample.role or "", order=order)
    elif sample.task == "kb_select":
        prompt = render_kb(sample.state_text, sample.options, role=sample.role or "", order=order)
    else:
        raise ValueError(f"not a pointer task: {sample.task}")

    tok = _tokenize(tokenizer, prompt.text, max_len)
    if tok is None:
        return None
    ids, offsets = tok
    positions = char_spans_to_token_indices(offsets, prompt.option_spans)
    if len(positions) != prompt.num_options:
        raise ValueError(f"{sample.id}: resolved {len(positions)} positions for {prompt.num_options} slots")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{sample.id}: two option lines resolved to the same token")

    enc = Encoded(
        task=sample.task,
        record_id=sample.id,
        role=sample.role,
        input_ids=ids,
        prompt_len=len(ids),
        q_position=len(ids) - 1,
        option_positions=positions,
        option_ids=list(prompt.option_ids),
        meta={"text": prompt.text, "canonical_indices": list(prompt.canonical_indices), "sample": sample.meta},
    )
    if sample.task == "tool_select":
        gold = str((sample.label or {}).get("id"))
        if gold == NO_TOOL:
            slot = int(prompt.meta["no_tool_slot"])
        else:
            canonical = [o.id for o in sample.options].index(gold)
            slot = prompt.canonical_indices.index(canonical)
        if prompt.option_ids[slot] != gold:
            raise AssertionError(f"{sample.id}: shuffled label mismatch {prompt.option_ids[slot]!r} != {gold!r}")
        enc.tool_label = slot
    else:
        grades = (sample.label or {}).get("grades") or {}
        soft = [float(grades.get(oid, 0)) / 2.0 for oid in prompt.option_ids]
        enc.kb_soft = soft
    return enc


def encode_context(sample: Sample, tokenizer: Any, max_len: int) -> Encoded | None:
    prompt = render_context(sample.state_text, sample.groups)
    tok = _tokenize(tokenizer, prompt.text, max_len)
    if tok is None:
        return None
    ids, offsets = tok
    positions = char_spans_to_token_indices(offsets, prompt.spans)
    if len(positions) != len(sample.groups):
        raise ValueError(f"{sample.id}: group span mismatch")
    labels = [CONTEXT_INDEX[str(x["label"])] for x in sample.label]
    return Encoded(
        task=sample.task,
        record_id=sample.id,
        role=None,
        input_ids=ids,
        prompt_len=len(ids),
        q_position=len(ids) - 1,
        group_positions=positions,
        group_labels=labels,
        meta={"text": prompt.text, "sample": sample.meta, "group_seqs": [int(g["seq"]) for g in sample.groups]},
    )


def encode_classification(sample: Sample, tokenizer: Any, max_len: int) -> Encoded | None:
    if sample.task == "role_select":
        prompt = render_role(sample.state_text)
        label = ROLE_INDEX[str((sample.label or {})["role"])]
        reason = None
    elif sample.task == "relay_select":
        prompt = render_relay(sample.from_role or "", sample.to_role or "", sample.why, sample.body)
        label = VERDICT_INDEX[str((sample.label or {})["verdict"])]
        reason = None
    elif sample.task == "premise_judge":
        prompt = render_premise(sample.unit, sample.left, sample.right, sample.context)
        label = PREMISE_INDEX[str((sample.label or {})["relation"])]
        reason = None
    elif sample.task == "overclaim_flag":
        prompt = render_overclaim(sample.left, sample.right, sample.context)
        label = None
        reason = None
    else:
        raise ValueError(f"not a classification task: {sample.task}")
    tok = _tokenize(tokenizer, prompt.text, max_len)
    if tok is None:
        return None
    ids, _ = tok
    flag_label = None
    if sample.task == "overclaim_flag":
        active = set((sample.label or {}).get("flags") or [])
        flag_label = [1.0 if f in active else 0.0 for f in OVERCLAIM_FLAGS]
    return Encoded(
        task=sample.task,
        record_id=sample.id,
        role=None,
        input_ids=ids,
        prompt_len=len(ids),
        q_position=len(ids) - 1,
        class_label=label,
        reason_label=reason,
        flag_label=flag_label,
        meta={"text": prompt.text, "sample": sample.meta},
    )


def encode_sample(
    sample: Sample,
    tokenizer: Any,
    max_len: int,
    order: Sequence[int] | None = None,
) -> Encoded | None:
    if sample.task in {"tool_select", "kb_select"}:
        return encode_pointer(sample, tokenizer, max_len, order=order)
    if sample.task == "context_retention":
        return encode_context(sample, tokenizer, max_len)
    return encode_classification(sample, tokenizer, max_len)


def batch_key(enc: Encoded) -> tuple:
    """Homogeneous batch bucket."""
    if enc.task in {"tool_select", "kb_select"}:
        return (enc.task, enc.role, enc.num_slots)
    if enc.task == "context_retention":
        return (enc.task, 0, 0)
    return (enc.task, 0, 0)


def pad_batch(examples: Sequence[Encoded], pad_token_id: int, device: str | None = None) -> dict:
    import torch

    if not examples:
        raise ValueError("empty batch")
    if len({batch_key(e) for e in examples}) != 1:
        raise ValueError(f"cannot mix batch keys in one batch: {[batch_key(e) for e in examples]}")
    task = examples[0].task
    bsz = len(examples)
    max_len = max(e.prompt_len for e in examples)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention = torch.zeros((bsz, max_len), dtype=torch.long)
    q_positions = torch.zeros(bsz, dtype=torch.long)
    for i, e in enumerate(examples):
        pad = max_len - e.prompt_len
        input_ids[i, pad:] = torch.tensor(e.input_ids, dtype=torch.long)
        attention[i, pad:] = 1
        q_positions[i] = e.q_position + pad
    batch: dict = {
        "task": task,
        "input_ids": input_ids,
        "attention_mask": attention,
        "q_positions": q_positions,
        "record_ids": [e.record_id for e in examples],
    }

    if task in {"tool_select", "kb_select"}:
        n_slots = examples[0].num_slots
        option_positions = torch.zeros((bsz, n_slots), dtype=torch.long)
        slot_mask = torch.ones((bsz, n_slots), dtype=torch.bool)
        for i, e in enumerate(examples):
            pad = max_len - e.prompt_len
            option_positions[i] = torch.tensor(e.option_positions, dtype=torch.long) + pad
        batch["option_positions"] = option_positions
        batch["slot_mask"] = slot_mask
        if task == "tool_select":
            batch["labels"] = torch.tensor([e.tool_label for e in examples], dtype=torch.long)
        else:
            batch["kb_soft"] = torch.tensor([e.kb_soft for e in examples], dtype=torch.float32)
        batch["option_ids"] = [e.option_ids for e in examples]
    elif task == "context_retention":
        max_groups = max(e.num_groups for e in examples)
        group_positions = torch.zeros((bsz, max_groups), dtype=torch.long)
        group_mask = torch.zeros((bsz, max_groups), dtype=torch.bool)
        labels = torch.full((bsz, max_groups), -100, dtype=torch.long)
        for i, e in enumerate(examples):
            pad = max_len - e.prompt_len
            n = e.num_groups
            group_positions[i, :n] = torch.tensor(e.group_positions, dtype=torch.long) + pad
            group_mask[i, :n] = True
            labels[i, :n] = torch.tensor(e.group_labels, dtype=torch.long)
        batch["group_positions"] = group_positions
        batch["group_mask"] = group_mask
        batch["labels"] = labels
    elif task == "overclaim_flag":
        batch["labels"] = torch.tensor([e.flag_label for e in examples], dtype=torch.float32)
    else:
        batch["labels"] = torch.tensor([e.class_label for e in examples], dtype=torch.long)

    if device:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return batch
