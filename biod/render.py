"""Prompt rendering for the decision model.

Three modes:

``choice`` (legacy, zero-training)
    One forward pass, listwise. Options are lettered ``A.``, ``B.``, ... and an
    optional ``NO_TOOL`` fallback is appended last.

``pointwise`` (legacy, zero-training)
    One forward pass per candidate, absolute score.

``pointer`` (SFT v0, trained heads)
    One forward pass over *all* candidates. Each candidate line is a numbered
    row; the trained pointer head needs the hidden state of the **last token of
    every option line**, so :func:`render_tool_prompt` / :func:`render_kb_prompt`
    also return the *character spans* of those lines. The spans are converted to
    token indices with :func:`char_spans_to_token_indices` using the tokenizer's
    ``offset_mapping``. Training and evaluation share this renderer, so the two
    never drift.

Contract (docs/SFT-v0-执行契约.md §3) tool template::

    <state>{state_text}</state>
    <question type="tool_choice">从下列能力中选一个最该加载的；没有适用项选 NO_TOOL。</question>
    <options>
    1. <name> — <description>
    ...
    0. NO_TOOL — 以上都不合适
    </options>
    <answer>

The knowledge-base template is isomorphic (``q_*``/``entry`` wording), with 32
numbered rows and no NO_TOOL row (sigmoid per-entry scoring, empty set possible).

The rendered text is intentionally plain and stable so that training-time and
inference-time distributions match.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .catalog import Option

LETTERS = [chr(ord("A") + i) for i in range(26)]

NO_TOOL_NAME = "NO_TOOL"
NO_TOOL_DESCRIPTION = "none of the above is appropriate; do not load anything"
NO_TOOL_ID = "NO_TOOL"
NO_TOOL_ROW_TEXT = "NO_TOOL — 以上都不合适"

CHOICE_TEMPLATE = """<state>
{state}
</state>
<question type="choice">
{instruction}
<options>
{options}
</options>
</question>
<answer>"""

POINTWISE_TEMPLATE = """<state>
{state}
</state>
<question type="yes_no">
{instruction}
<candidate>
{name} — {description}
</candidate>
Is this candidate directly useful for the need described in <state>? Answer Yes or No.
</question>
<answer>"""

# Pointer template: shared by the tool and knowledge-base heads. The only
# differences are the question ``type`` and the instruction line.
POINTER_TEMPLATE = """<state>
{state}
</state>
<question type="{qtype}">
{instruction}
<options>
{options}
</options>
</question>
<answer>"""

DEFAULT_CHOICE_INSTRUCTION = "Select exactly one capability to load next. Choose NO_TOOL if none applies."
DEFAULT_SCORE_INSTRUCTION = "Judge whether the candidate knowledge entry is useful for the current need."
DEFAULT_TOOL_POINTWISE_INSTRUCTION = "Judge whether the candidate capability is directly useful for the current need."

# v0 SFT instructions (Chinese, matching the zh-dominant eval distribution).
TOOL_POINTER_INSTRUCTION = "从下列能力中选一个最该加载的；没有适用项选 NO_TOOL。"
KB_POINTER_INSTRUCTION = "从下列知识条目中选出与需求相关的条目（可多选；若都不相关则都不选）。"

TOOL_QTYPE = "tool_choice"
KB_QTYPE = "knowledge_score"


# --------------------------------------------------------------------------- #
# legacy API (kept unchanged for zero-training / regression tests)
# --------------------------------------------------------------------------- #
def render_choice(
    state: str,
    options: Sequence[Option],
    instruction: str = DEFAULT_CHOICE_INSTRUCTION,
    add_no_tool: bool = True,
) -> tuple[str, list[str], list[str]]:
    """Render a listwise choice prompt.

    Returns ``(prompt, labels, ids)`` where ``labels`` are the letter labels in
    order and ``ids`` are the option ids (``NO_TOOL`` id is ``__no_tool__``).
    """
    if len(options) > 26:
        raise ValueError(
            f"choice rendering supports at most 26 options, got {len(options)}; "
            "use pointwise rendering or a hierarchical layer"
        )
    lines: list[str] = []
    labels: list[str] = []
    ids: list[str] = []
    for i, opt in enumerate(options):
        letter = LETTERS[i]
        labels.append(letter)
        ids.append(opt.id)
        lines.append(f"{letter}. {opt.render_text()}")
    if add_no_tool:
        letter = LETTERS[len(options)]
        labels.append(letter)
        ids.append("__no_tool__")
        lines.append(f"{letter}. {NO_TOOL_NAME} — {NO_TOOL_DESCRIPTION}")
    prompt = CHOICE_TEMPLATE.format(
        state=state.strip(),
        instruction=instruction.strip(),
        options="\n".join(lines),
    )
    return prompt, labels, ids


def render_pointwise(
    state: str,
    option: Option,
    instruction: str = DEFAULT_SCORE_INSTRUCTION,
) -> str:
    """Render a single-candidate yes/no prompt."""
    return POINTWISE_TEMPLATE.format(
        state=state.strip(),
        instruction=instruction.strip(),
        name=option.name,
        description=option.description,
    )


def render_domain_choice(state: str, domains: Iterable, instruction: str) -> tuple[str, list[str], list[str]]:
    """Render level-1 domain choice using the same choice template."""
    from .catalog import domain_as_options

    return render_choice(state, domain_as_options(domains, layer="tool_l1"), instruction=instruction)


# --------------------------------------------------------------------------- #
# pointer mode (SFT v0): text + option-line char spans
# --------------------------------------------------------------------------- #
@dataclass
class PointerPrompt:
    """Rendered pointer prompt.

    Attributes
    ----------
    text:
        Full prompt string.
    option_spans:
        ``[(char_start, char_end), ...]`` for every option line (line content
        only, no trailing newline), in display order. ``option_spans[i]`` maps
        to slot ``i`` of the pointer head.
    option_ids:
        Candidate id per display slot. For the tool task the last slot is the
        ``NO_TOOL`` row (id ``"NO_TOOL"``); for the knowledge base there is no
        NO_TOOL row.
    canonical_indices:
        For each display slot, the canonical option index it was rendered from
        (``-1`` for the synthetic ``NO_TOOL`` row). Used to map shuffled labels
        back and forth.
    qtype / instruction:
        Template parameters, recorded for reproducibility.
    """

    text: str
    option_spans: list[tuple[int, int]]
    option_ids: list[str]
    canonical_indices: list[int]
    qtype: str
    instruction: str
    state: str = ""
    state_char_span: tuple | None = None  # char span of the state text (for q=state_last)
    meta: dict = field(default_factory=dict)

    @property
    def num_options(self) -> int:
        return len(self.option_spans)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "option_spans": [list(s) for s in self.option_spans],
            "option_ids": list(self.option_ids),
            "canonical_indices": list(self.canonical_indices),
            "qtype": self.qtype,
            "instruction": self.instruction,
        }


def _render_pointer(
    state: str,
    rows: Sequence[str],
    qtype: str,
    instruction: str,
) -> tuple[str, list[tuple[int, int]], tuple[int, int]]:
    """Assemble the pointer template and record char spans of option lines."""
    state_clean = state.strip()
    state_prefix = "<state>\n"
    head = (
        f"{state_prefix}{state_clean}\n</state>\n"
        f'<question type="{qtype}">\n{instruction.strip()}\n<options>\n'
    )
    state_span = (len(state_prefix), len(state_prefix) + len(state_clean))
    offset = len(head)
    spans: list[tuple[int, int]] = []
    for row in rows:
        spans.append((offset, offset + len(row)))
        offset += len(row) + 1  # +1 for the newline joining rows
    tail = "\n</options>\n</question>\n<answer>"
    text = head + "\n".join(rows) + tail
    # sanity: every recorded span must slice back to the original row
    for (s, e), row in zip(spans, rows):
        if text[s:e] != row:
            raise AssertionError("option char span misaligned; renderer bug")
    return text, spans, state_span


def canonical_order(n: int) -> list[int]:
    """Canonical display order: option index ``i`` rendered at slot ``i``."""
    return list(range(n))


def reversed_order(n: int) -> list[int]:
    """Reversed display order (used by eval for order-flip rate)."""
    return list(range(n - 1, -1, -1))


def render_tool_prompt(
    state: str,
    options: Sequence[Option],
    order: Sequence[int] | None = None,
    instruction: str = TOOL_POINTER_INSTRUCTION,
) -> PointerPrompt:
    """Render the tool-choice pointer prompt.

    ``order`` is a sequence of canonical option indices giving the display order
    (``None`` = canonical). Rows are numbered ``1..N`` by display position; the
    trailing ``NO_TOOL`` row is numbered ``0`` and is always last, so the
    pointer has ``len(options) + 1`` slots.
    """
    n = len(options)
    if order is None:
        order = canonical_order(n)
    order = list(order)
    if sorted(order) != list(range(n)):
        raise ValueError(f"order must be a permutation of 0..{n - 1}, got {order}")
    rows: list[str] = []
    for display_pos, canonical_idx in enumerate(order):
        rows.append(f"{display_pos + 1}. {options[canonical_idx].render_text()}")
    rows.append(f"0. {NO_TOOL_ROW_TEXT}")
    text, spans, state_span = _render_pointer(state, rows, TOOL_QTYPE, instruction)
    ids = [options[i].id for i in order] + [NO_TOOL_ID]
    return PointerPrompt(
        text=text,
        option_spans=spans,
        option_ids=ids,
        canonical_indices=order + [-1],
        qtype=TOOL_QTYPE,
        instruction=instruction,
        state=state,
        state_char_span=state_span,
        meta={"no_tool_slot": len(options)},
    )


def render_kb_prompt(
    state: str,
    options: Sequence[Option],
    order: Sequence[int] | None = None,
    instruction: str = KB_POINTER_INSTRUCTION,
) -> PointerPrompt:
    """Render the knowledge-base pointer prompt (one row per entry, no NO_TOOL)."""
    n = len(options)
    if order is None:
        order = canonical_order(n)
    order = list(order)
    if sorted(order) != list(range(n)):
        raise ValueError(f"order must be a permutation of 0..{n - 1}, got {order}")
    rows = [f"{display_pos + 1}. {options[canonical_idx].render_text()}" for display_pos, canonical_idx in enumerate(order)]
    text, spans, state_span = _render_pointer(state, rows, KB_QTYPE, instruction)
    ids = [options[i].id for i in order]
    return PointerPrompt(
        text=text,
        option_spans=spans,
        option_ids=ids,
        canonical_indices=order,
        qtype=KB_QTYPE,
        instruction=instruction,
        state=state,
        state_char_span=state_span,
    )


# --------------------------------------------------------------------------- #
# char span -> token index
# --------------------------------------------------------------------------- #
def char_end_to_token_index(offset_mapping: Sequence[Sequence[int]], char_end: int) -> int:
    """Token index of the last token that starts before ``char_end``."""
    starts = [int(off[0]) for off in offset_mapping]
    idx = bisect_left(starts, char_end) - 1
    if idx < 0:
        raise ValueError(f"no token before char end {char_end}")
    return idx


def char_spans_to_token_indices(
    offset_mapping: Sequence[Sequence[int]],
    char_spans: Sequence[Sequence[int]],
) -> list[int]:
    """Map option-line char spans to the token index of each line's last token.

    ``offset_mapping`` comes from a fast tokenizer called with
    ``add_special_tokens=False``. Option lines never contain newlines, so the
    last token of a line is simply the last token whose start offset is before
    the line end: ``bisect_left(starts, char_end) - 1``.

    Raises ``ValueError`` if a span cannot be resolved (rather than silently
    returning a wrong position).
    """
    starts = [int(off[0]) for off in offset_mapping]
    ends = [int(off[1]) for off in offset_mapping]
    out: list[int] = []
    for span in char_spans:
        start, end = int(span[0]), int(span[1])
        idx = bisect_left(starts, end) - 1
        if idx < 0:
            raise ValueError(f"no token before char end {end}")
        # Guard against a token starting before the line (pathological merge of
        # newline + first line char). Fall back to the last token starting
        # inside the span.
        if starts[idx] < start:
            inside = [i for i in range(len(starts)) if start <= starts[i] < end]
            if not inside:
                raise ValueError(
                    f"cannot resolve option span {(start, end)} to a token; "
                    f"nearest token start={starts[idx]}"
                )
            idx = inside[-1]
        if ends[idx] <= starts[idx]:
            raise ValueError(f"zero-width token at index {idx}")
        out.append(idx)
    return out
