"""Prompt renderers for the five decisions.

Tool/kb reuse the v0 pointer templates (``biod.render``) so the learned head
sees the same layout as the existing pipeline. The three new tasks are
section renders: one prompt, one readout position (classification) or one span
per group (3-way disposition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from biod.render import (
    PointerPrompt,
    render_kb_prompt,
    render_tool_prompt,
)

from .catalog import CatalogOption
from .tasks import CONTEXT_LABELS, OVERCLAIM_FLAGS, PREMISE_LABELS, RELAY_VERDICTS, ROLES


TOOL_INSTRUCTION = (
    "从下列能力中选一个下一步最该加载的；没有适用项选 NO_TOOL。"
    "只选一个，不要输出候选之外的 id。"
)
KB_INSTRUCTION = (
    "判断下列知识条目对当前这一步是否有用；可以多选，也可以一条都不选。"
    "只输出候选中的 id。"
)
CONTEXT_INSTRUCTION = "对下面每一组历史：原文保留（keep）/ 摘要（summarise）/ 丢弃（drop）？"
ROLE_POLICY = (
    "如果一个非 PI 角色能独立完成，就选它；需要跨角色调度、需要验收/仲裁、"
    "全流程项目、或你不确定时，选 pi。只输出一个角色。"
)
RELAY_POLICY = (
    "allow = 这一跳必要、目标角色负责、无重复劳动/越权；"
    "reject = 不必要/目标不负责/重复劳动/让目标做越权决策；"
    "escalate = 语义模糊、高风险、需要 PI 仲裁、或你不确定。"
)

ROLE_ROSTER = {
    "pi": "研究目标、任务分配、决策与成果验收；不替其他角色做具体工作。",
    "searcher": "文献检索与证据综合、研究现状、假设生成与比较；不执行实验、不写正文。",
    "researcher": "研究方案与实验设计、方法选择、执行、统计、结果解释；不做文献综述、不写正文。",
    "writer": "论文框架与叙事、结果组织、图表、引用、投稿材料；不改数据、不跑分析。",
    "reviewer": "创新性、假设、方法学、数据与统计、结论一致性审查；不直接改稿。",
}


@dataclass
class SectionPrompt:
    """A rendered classification prompt.

    ``spans`` is used by the group-3 head (char span per group, in display
    order). For role/relay it is empty and the readout is the last token.
    """

    text: str
    spans: list[tuple[int, int]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def render_tool(state: str, options: Sequence[CatalogOption], role: str = "", order: Sequence[int] | None = None) -> PointerPrompt:
    if len(state) > 800:
        state = state[:800].rstrip() + "…"
    if role:
        state = f"[角色：{role}]\n{state}"
    return render_tool_prompt(state, options, order=order, instruction=TOOL_INSTRUCTION)


def render_kb(state: str, options: Sequence[CatalogOption], role: str = "", order: Sequence[int] | None = None) -> PointerPrompt:
    if len(state) > 1600:
        state = state[:1600].rstrip() + "…"
    if role:
        state = f"[角色：{role}]\n{state}"
    return render_kb_prompt(state, options, order=order, instruction=KB_INSTRUCTION)


def render_context(state: str, groups: Sequence[dict]) -> SectionPrompt:
    """Render one cold-zone batch; one char span per group (its text)."""
    lines: list[str] = ["<task>context_retention</task>", "<state>", state.strip(), "</state>", "<groups>"]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for group in groups:
        header = f"[seq {int(group['seq'])}] {group.get('kind', 'tool')} / {group.get('tool', '') or 'n/a'} / {group.get('outcome', 'unknown')}"
        lines.append(header)
        text = str(group.get("text", "")).strip()
        # Span covers the human-readable group text; the header line above it
        # carries the seq/kind and is not part of the readout.
        prefix = "\n".join(lines) + "\n"
        start = len(prefix)
        end = start + len(text)
        spans.append((start, end))
        lines.append(text)
        cursor = end
    lines.extend(["</groups>", f"<question>{CONTEXT_INSTRUCTION}</question>", "<answer>"])
    text = "\n".join(lines)
    # Recompute spans against the final text (the loop above already used the
    # exact same prefix construction, so they match; assert to catch drift).
    assert cursor <= len(text)
    return SectionPrompt(text=text, spans=spans, meta={"labels": list(CONTEXT_LABELS)})


def render_role(state: str) -> SectionPrompt:
    roster = "\n".join(f"- {rid}：{ROLE_ROSTER[rid]}" for rid in ROLES)
    text = (
        "<task>role_select</task>\n"
        "<team>\n"
        f"{roster}\n"
        "</team>\n"
        f"<policy>{ROLE_POLICY}</policy>\n"
        "<state>\n"
        f"{state.strip()}\n"
        "</state>\n"
        "<answer>"
    )
    return SectionPrompt(text=text, meta={"labels": list(ROLES)})


PREMISE_INSTRUCTION = (
    "判断 <right> 对 <left> 的关系，只输出一个标签："
    "supports（同一对象/范围/测量上直接支持）/ partially_supports（方向一致但覆盖不全）/ "
    "contradicts（同一关系上给出相反结果）/ unrelated（同领域但不构成证据）/ "
    "insufficient（谈的是同一问题但无法判定）。"
)
PREMISE_UNIT_TEXT = {
    "hypothesis": "研究假设",
    "claim": "论文主张",
    "issue": "审稿 issue 的前提",
}
OVERCLAIM_INSTRUCTION = (
    "标出 <claim> 中超出 <evidence> 支持范围的声称类型，可多选，也可为空："
    "scope（外推到未测试范围）/ causality（相关性说成因果）/ certainty（确定性超出证据强度）/ "
    "generalization（单一数据集泛化为普遍规律）/ temporal（无时间序列证据却称长期有效）。"
)


def render_premise(unit: str, left: str, right: str, context: str = "") -> SectionPrompt:
    head = f"<task>premise_judge</task>\n<unit>{PREMISE_UNIT_TEXT.get(unit, unit)}</unit>\n"
    if context.strip():
        head += f"<context>\n{context.strip()}\n</context>\n"
    text = (
        head
        + "<left>\n" + left.strip() + "\n</left>\n"
        + "<right>\n" + right.strip() + "\n</right>\n"
        + f"<question>{PREMISE_INSTRUCTION}</question>\n<answer>"
    )
    return SectionPrompt(text=text, meta={"labels": list(PREMISE_LABELS)})


def render_overclaim(claim: str, evidence: str, context: str = "") -> SectionPrompt:
    head = "<task>overclaim_flag</task>\n"
    if context.strip():
        head += f"<context>\n{context.strip()}\n</context>\n"
    text = (
        head
        + "<claim>\n" + claim.strip() + "\n</claim>\n"
        + "<evidence>\n" + evidence.strip() + "\n</evidence>\n"
        + f"<question>{OVERCLAIM_INSTRUCTION}</question>\n<answer>"
    )
    return SectionPrompt(text=text, meta={"labels": list(OVERCLAIM_FLAGS)})


def render_relay(from_role: str, to_role: str, why: str, body: str) -> SectionPrompt:
    text = (
        "<task>relay_select</task>\n"
        f"<from>{from_role}</from>\n"
        f"<to>{to_role}</to>\n"
        f"<why>{why.strip()}</why>\n"
        "<body>\n"
        f"{body.strip()}\n"
        "</body>\n"
        f"<policy>{RELAY_POLICY}</policy>\n"
        "<answer>"
    )
    return SectionPrompt(text=text, meta={"labels": list(RELAY_VERDICTS)})
