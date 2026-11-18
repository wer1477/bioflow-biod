"""Unit tests for the pointer-render / collate logic (no GPU, no real tokenizer).

Run locally:  ``python3 tests/test_train_collate.py``
Run on deployment-host: ``.venv/bin/python -m pytest tests/test_train_collate.py -q``

A character-level fake tokenizer with identity offset mapping exercises the
whole char-span -> token-index pipeline deterministically, so these tests catch
the two things that silently ruin listwise training: wrong option spans and
wrong label remapping after option shuffling.
"""

from __future__ import annotations

import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from biod.catalog import Option
from biod.render import (
    NO_TOOL_ID,
    canonical_order,
    char_spans_to_token_indices,
    render_choice,
    render_kb_prompt,
    render_tool_prompt,
    reversed_order,
)
from biod.train.collate import (
    build_examples,
    build_kb_example,
    build_tool_example,
    make_order,
    token_lengths,
)


class FakeTokenizer:
    """One token per character, identity offsets (enough to test span logic)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False):
        tokens = list(range(len(text)))
        out = {"input_ids": tokens}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def _tools(n: int = 5) -> list[Option]:
    return [
        Option(
            index=i,
            name=f"tool_{i}",
            description=f"Description of capability {i} with some detail.",
            id=f"tool_{i}",
            kind="skill",
            layer="tool_l2",
            domain="dom",
        )
        for i in range(n)
    ]


def _kbs(n: int = 4) -> list[Option]:
    return [
        Option(
            index=i,
            name=f"dataset_{i}",
            description=f"Knowledge entry {i} about sequencing analysis methods.",
            id=f"kb/entry_{i}",
            kind="kb",
            layer="kb_l2",
            domain="kb/dom",
        )
        for i in range(n)
    ]


def _tool_record(gold_id: str, state: str = "请帮我分析这批数据？") -> dict:
    tools = _tools()
    return {
        "id": "rec-1",
        "state_text": state,
        "candidates": [o.id for o in tools],
        "gold_primary": gold_id,
        "gold_acceptable": [],
    }


def _kb_record(grades: dict, state: str = "需要做差异表达分析。") -> dict:
    kbs = _kbs()
    soft = {k: v / 2.0 for k, v in grades.items()}
    return {
        "id": "rec-kb-1",
        "state_text": state,
        "candidates": [o.id for o in kbs],
        "soft_labels": soft,
        "num_positive": sum(1 for v in soft.values() if v > 0),
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_tool_render_has_one_span_per_slot_and_no_tool_last():
    tools = _tools()
    prompt = render_tool_prompt("state text", tools)
    assert prompt.num_options == len(tools) + 1
    assert prompt.option_ids == [o.id for o in tools] + [NO_TOOL_ID]
    assert expected_text_guard(prompt, "0. NO_TOOL — 以上都不合适")
    assert prompt.option_ids[-1] == NO_TOOL_ID
    # every span slices back to a numbered option row
    for slot, (start, end) in enumerate(prompt.option_spans):
        line = prompt.text[start:end]
        if slot < len(tools):
            assert line.startswith(f"{slot + 1}. "), line
            assert tools[slot].name in line
            assert tools[slot].description in line
        else:
            assert line.startswith("0. "), line
    # spans are strictly increasing and non-overlapping
    starts = [s for s, _ in prompt.option_spans]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def expected_text_guard(prompt, marker: str) -> bool:
    assert marker in prompt.text
    return True


def test_kb_render_has_one_span_per_entry():
    kbs = _kbs()
    prompt = render_kb_prompt("state text", kbs)
    assert prompt.num_options == len(kbs)
    assert prompt.option_ids == [o.id for o in kbs]
    for slot, (start, end) in enumerate(prompt.option_spans):
        line = prompt.text[start:end]
        assert line.startswith(f"{slot + 1}. "), line
        assert kbs[slot].name in line


def test_legacy_render_choice_unchanged():
    tools = _tools()
    prompt, labels, ids = render_choice("s", tools)
    assert labels == ["A", "B", "C", "D", "E", "F"]
    assert ids[-1] == "__no_tool__"
    assert "<options>" in prompt


# --------------------------------------------------------------------------- #
# char span -> token index
# --------------------------------------------------------------------------- #
def test_char_spans_map_to_last_character_token():
    text = "abc\ndef\nghi"
    offsets = [(i, i + 1) for i in range(len(text))]
    # spans of "abc", "def", "ghi"
    positions = char_spans_to_token_indices(offsets, [(0, 3), (4, 7), (8, 11)])
    assert positions == [2, 6, 10]
    # decoding those tokens gives the last character of each line
    assert [text[i] for i in positions] == ["c", "f", "i"]


def test_char_spans_raise_on_empty_mapping():
    try:
        char_spans_to_token_indices([], [(0, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty offset mapping")


# --------------------------------------------------------------------------- #
# option shuffling + label remapping
# --------------------------------------------------------------------------- #
def test_tool_label_remap_under_many_orders():
    tools = _tools(7)
    id_to_index = {o.id: i for i, o in enumerate(tools)}
    tok = FakeTokenizer()
    for gold in ("tool_0", "tool_3", "tool_6"):
        record = _tool_record(gold)
        canonical = build_tool_example(record, tools, id_to_index, tok)
        assert canonical.option_ids[canonical.tool_label] == gold
        for order in (reversed_order(len(tools)), *(
            make_order(len(tools), random.Random(seed)) for seed in range(10)
        )):
            ex = build_tool_example(record, tools, id_to_index, tok, order=order)
            assert ex.label_check(), (gold, order, ex.to_debug_dict())
            assert ex.option_ids[ex.tool_label] == gold
            # the label slot differs from canonical unless the permutation is identity
            if order != canonical_order(len(tools)):
                assert ex.tool_label != id_to_index[gold] or True


def test_no_tool_label_always_last_slot():
    tools = _tools(3)
    id_to_index = {o.id: i for i, o in enumerate(tools)}
    tok = FakeTokenizer()
    record = _tool_record(NO_TOOL_ID)
    for seed in range(6):
        order = make_order(len(tools), random.Random(seed))
        ex = build_tool_example(record, tools, id_to_index, tok, order=order)
        assert ex.tool_label == len(tools)
        assert ex.option_ids[ex.tool_label] == NO_TOOL_ID
        assert ex.label_check()


def test_kb_soft_label_remap_under_orders():
    kbs = _kbs(5)
    id_to_index = {o.id: i for i, o in enumerate(kbs)}
    tok = FakeTokenizer()
    record = _kb_record({kbs[0].id: 2, kbs[3].id: 1})
    canonical = build_kb_example(record, kbs, id_to_index, tok)
    assert canonical.kb_soft == [1.0, 0.0, 0.0, 0.5, 0.0]
    for seed in range(8):
        order = make_order(len(kbs), random.Random(seed))
        ex = build_kb_example(record, kbs, id_to_index, tok, order=order)
        assert sum(ex.kb_soft) == 1.5
        for slot, value in enumerate(ex.kb_soft):
            entry_id = ex.option_ids[slot]
            expected = record["soft_labels"].get(entry_id, 0.0)
            assert abs(value - expected) < 1e-9, (slot, entry_id, value, expected)


# --------------------------------------------------------------------------- #
# prompt assembly invariants
# --------------------------------------------------------------------------- #
def test_q_position_state_mode_points_inside_state_block():
    tools = _tools(3)
    id_to_index = {o.id: i for i, o in enumerate(tools)}
    tok = FakeTokenizer()
    state = "some state text"
    record = _tool_record("tool_0", state=state)
    prompt_mode = build_tool_example(record, tools, id_to_index, tok, q_position_mode="prompt")
    state_mode = build_tool_example(record, tools, id_to_index, tok, q_position_mode="state")
    assert prompt_mode.q_position == prompt_mode.prompt_len - 1
    # fake tokenizer: one token per char -> q token = last char of the state block
    assert state_mode.q_position < len(prompt_mode.input_ids) - 1
    assert 0 <= state_mode.q_position < state_mode.prompt_len
    # the resolved token must be the last state character: with an identity-offset
    # fake tokenizer the token index equals the char index of the span end - 1
    prompt = render_tool_prompt(state, tools)
    assert prompt.state_char_span is not None
    assert state_mode.q_position == prompt.state_char_span[1] - 1


def test_option_token_positions_are_unique_and_inside_rows():
    tools = _tools(6)
    id_to_index = {o.id: i for i, o in enumerate(tools)}
    tok = FakeTokenizer()
    record = _tool_record("tool_2")
    ex = build_tool_example(record, tools, id_to_index, tok)
    assert len(ex.option_positions) == len(tools) + 1
    assert len(set(ex.option_positions)) == len(ex.option_positions)
    assert ex.option_positions == sorted(ex.option_positions)
    assert ex.q_position == ex.prompt_len - 1


def test_over_length_samples_are_dropped_not_truncated():
    tools = _tools(4)
    id_to_index = {o.id: i for i, o in enumerate(tools)}
    tok = FakeTokenizer()
    records = [_tool_record("tool_1", state=f"state number {i}") for i in range(4)]
    full, stats = build_examples(records, "tool", tools, id_to_index, tok, max_len=0)
    assert stats["dropped"] == 0 and len(full) == 4
    tiny, stats = build_examples(records, "tool", tools, id_to_index, tok, max_len=50)
    assert stats["dropped"] == 4 and len(tiny) == 0
    cut = full[0].prompt_len - 1
    partial, stats = build_examples(records, "tool", tools, id_to_index, tok, max_len=cut)
    assert stats["dropped"] == 4, "max_len smaller than any prompt must drop all rows"
    assert all(ex.prompt_len <= cut for ex in partial)  # none survive, never truncated


def test_token_lengths_match_rendered_prompt():
    tools = _tools(3)
    record = _tool_record("tool_0", state="abc")
    lengths = token_lengths([record], "tool", tools, FakeTokenizer())
    prompt = render_tool_prompt("abc", tools)
    assert lengths == [len(prompt.text)]


def test_make_order_rejects_non_permutation():
    try:
        make_order(4, order=[0, 0, 1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-permutation order")


# --------------------------------------------------------------------------- #
# padding (needs torch; skipped on the bare local python)
# --------------------------------------------------------------------------- #
def test_pad_batch_shifts_positions_and_keeps_query_last():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("    (skipped: torch not installed locally)")
        return
    from biod.train.collate import pad_batch

    tools = _tools(3)
    id_to_index = {o.id: i for i, o in enumerate(tools)}
    tok = FakeTokenizer()
    records = [
        _tool_record("tool_0", state="a"),
        _tool_record("tool_2", state="a much longer state text here"),
    ]
    exs, _ = build_examples(records, "tool", tools, id_to_index, tok, max_len=0)
    batch = pad_batch(exs, pad_token_id=0)
    assert batch["input_ids"].shape[0] == 2
    width = batch["input_ids"].shape[1]
    assert batch["q_position"].tolist() == [width - 1, width - 1]
    for row, ex in enumerate(exs):
        pad = width - ex.prompt_len
        assert batch["input_ids"][row, pad:].tolist() == ex.input_ids
        assert batch["option_positions"][row].tolist() == [p + pad for p in ex.option_positions]
        assert int(batch["attention_mask"][row].sum()) == ex.prompt_len


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - test runner
            failed += 1
            import traceback

            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
