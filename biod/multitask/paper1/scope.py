"""Paper 1 E1 scope manipulation on frozen test items.

This module keeps the label fixed and changes only what the decision observes:
local paired evidence vs global team-state/dossier, plus two length controls.
The goal is a clean within-item contrast, not a new dataset.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Iterable, Sequence

from transformers import AutoTokenizer

from biod.render import render_kb_prompt, render_tool_prompt
from ..render import CONTEXT_INSTRUCTION, RELAY_POLICY, ROLE_POLICY
from ..render import ROLE_ROSTER as ROLE_ROSTER_DICT

from ..catalog import CatalogSnapshot, load_catalog_snapshot
from ..collate import Encoded, _tokenize, char_spans_to_token_indices
from ..data import Sample, load_split
from ..infer import predict_encoded
from ..evaluate import compute
from ..tasks import TASK_SPECS, TASKS

ROLE_ROSTER = (
    "pi 负责目标、任务分配、验收与仲裁；searcher 负责文献检索与证据综合；"
    "researcher 负责方案、实验与统计；writer 负责写作与图表；reviewer 负责审查。"
)

FILLER = (
    "（背景记录：本阶段不改变任务目标；所有判定只使用当前可见材料；"
    "历史记录已按时间顺序归档；如需复现，请保留来源、版本与校验和。）"
)


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def global_text(sample: Sample, task: str) -> str:
    """Synthesise a global team-state/dossier block, label-free by construction."""
    role = sample.role or "pi"
    stage = {
        "tool_select": "下一步能力选择",
        "kb_select": "知识检索与证据确认",
        "context_retention": "冷区上下文压缩",
        "role_select": "消息路由",
        "relay_select": "角色间转交门禁",
        "premise_judge": "证据-主张关系判定",
        "overclaim_flag": "主张-证据审计",
    }.get(task, "判定")
    if task in {"tool_select", "kb_select"}:
        goal = sample.state_text.strip()[:500]
        mats = "；".join(f"{o.name}: {o.description[:60]}" for o in sample.options[:8])
        tail = "候选能力由运行时给定。"
    elif task == "role_select":
        goal = sample.state_text.strip()[:500]
        mats = ROLE_ROSTER
        tail = "消息归属按角色职责判定。"
    elif task == "relay_select":
        goal = (sample.why or "").strip()[:400]
        mats = (sample.body or "").strip()[:600]
        tail = f"当前转交：{sample.from_role} -> {sample.to_role}。"
    elif task == "context_retention":
        goal = sample.state_text.strip()[:400]
        mats = "；".join(f"seq {g['seq']}: {str(g.get('text',''))[:80]}" for g in sample.groups[:6])
        tail = "只处理冷区历史。"
    elif task == "premise_judge":
        goal = (sample.left or "").strip()[:400]
        mats = (sample.right or "").strip()[:700]
        tail = "按证据关系判定。"
    elif task == "overclaim_flag":
        goal = (sample.left or "").strip()[:300]
        mats = (sample.right or "").strip()[:700]
        tail = "按证据支持范围判定。"
    else:
        goal = ""
        mats = ""
        tail = ""
    return (
        "[团队态]\n"
        f"当前主责：{role}\n"
        f"当前阶段：{stage}\n"
        "[项目背景]\n"
        f"目标：{goal}\n"
        f"已知材料：{mats}\n"
        "约束：只使用当前可见材料；不越过角色权限；保留可复现记录。\n"
        f"备注：{tail} 以上背景不改变任务规则与候选集合。\n"
    )


def _scope_value(sample: Sample, task: str, scope: str) -> str:
    if task in {"tool_select", "kb_select", "context_retention", "role_select"}:
        base = sample.state_text
    elif task == "relay_select":
        base = sample.why or ""
    else:  # premise_judge / overclaim_flag
        base = sample.context or ""
    g = global_text(sample, task)
    if scope == "local":
        return base
    if scope == "global":
        return f"{base}\n\n{g}"
    if scope == "local_len":
        target = len(base) + len(g)
        if len(base) >= target:
            return base
        reps = (target - len(base)) // len(FILLER) + 1
        return base + ("\n" + FILLER * reps)[: target - len(base)]
    if scope == "global_len":
        # content from the global block, clipped to the local length
        target = max(1, len(base))
        return g[:target]
    raise ValueError(scope)


def scope_sample(sample: Sample, task: str, scope: str) -> Sample:
    text = _scope_value(sample, task, scope)
    if task in {"tool_select", "kb_select", "context_retention", "role_select"}:
        return dataclasses.replace(sample, state_text=text)
    if task == "relay_select":
        return dataclasses.replace(sample, why=text)
    if task in {"premise_judge", "overclaim_flag"}:
        return dataclasses.replace(sample, context=text)
    raise ValueError(task)


def _parts(sample: Sample, task: str, scope: str) -> tuple[str, str]:
    """Return (local_carrier, context_block) for role/relay/context."""
    if task == "role_select":
        base = sample.state_text
    elif task == "relay_select":
        base = sample.body or ""
    elif task == "context_retention":
        base = sample.state_text
    else:
        raise ValueError(task)
    g = global_text(sample, task)
    if scope == "local":
        return base, ""
    if scope == "global":
        return base, f"<context>\n{g}\n</context>\n"
    if scope == "local_len":
        target = len(base) + len(g)
        need = max(0, target - len(base))
        reps = need // len(FILLER) + 1
        return base + ("\n" + FILLER * reps)[:need], ""
    if scope == "global_len":
        return base, f"<context>\n{g[:max(1, len(base))]}\n</context>\n"
    raise ValueError(scope)


def _render_role_scope(sample: Sample, scope: str) -> str:
    base, ctx = _parts(sample, "role_select", scope)
    roster = "\n".join(f"- {rid}：{ROLE_ROSTER_DICT[rid]}" for rid in ROLE_ROSTER_DICT)
    return (
        "<task>role_select</task>\n"
        "<team>\n" + roster + "\n</team>\n"
        f"<policy>{ROLE_POLICY}</policy>\n"
        f"{ctx}"
        "<state>\n" + base.strip() + "\n</state>\n"
        "<answer>"
    )


def _render_relay_scope(sample: Sample, scope: str) -> str:
    base, ctx = _parts(sample, "relay_select", scope)
    return (
        "<task>relay_select</task>\n"
        f"<from>{sample.from_role}</from>\n"
        f"<to>{sample.to_role}</to>\n"
        f"<why>{(sample.why or '').strip()}</why>\n"
        "<body>\n" + base.strip() + "\n</body>\n"
        f"{ctx}"
        f"<policy>{RELAY_POLICY}</policy>\n"
        "<answer>"
    )


def _render_context_scope(sample: Sample, scope: str) -> tuple[str, list[tuple[int, int]]]:
    base, ctx = _parts(sample, "context_retention", scope)
    lines: list[str] = ["<task>context_retention</task>"]
    if ctx:
        lines.extend(ctx.rstrip("\n").splitlines())
    lines.extend(["<state>", base.strip(), "</state>", "<groups>"])
    spans: list[tuple[int, int]] = []
    for group in sample.groups:
        header = f"[seq {int(group['seq'])}] {group.get('kind','tool')} / {group.get('tool','') or 'n/a'} / {group.get('outcome','unknown')}"
        lines.append(header)
        text = str(group.get("text", "")).strip()
        prefix = "\n".join(lines) + "\n"
        start = len(prefix)
        end = start + len(text)
        spans.append((start, end))
        lines.append(text)
    lines.extend(["</groups>", f"<question>{CONTEXT_INSTRUCTION}</question>", "<answer>"])
    return "\n".join(lines), spans


def encode_scope(sample: Sample, tokenizer, task: str, scope: str, catalog: CatalogSnapshot) -> Encoded | None:
    """Encode a scope variant. Pointer tasks bypass the 800/1600-char state truncation."""
    max_len = TASK_SPECS[task].max_len
    if task in {"tool_select", "kb_select"}:
        s = scope_sample(sample, task, scope)
        state = s.state_text
        if s.role:
            state = f"[角色：{s.role}]\n{state}"
        prompt = render_tool_prompt(state, s.options) if task == "tool_select" else render_kb_prompt(state, s.options)
        tok = _tokenize(tokenizer, prompt.text, max_len)
        if tok is None:
            return None
        ids, offsets = tok
        positions = char_spans_to_token_indices(offsets, prompt.option_spans)
        if len(positions) != prompt.num_options:
            raise ValueError(f"{s.id}: option span mismatch")
        enc = Encoded(task=task, record_id=f"{s.id}::{scope}", role=s.role, input_ids=ids,
                      prompt_len=len(ids), q_position=len(ids) - 1,
                      option_positions=positions, option_ids=list(prompt.option_ids),
                      meta={"text": prompt.text, "base_id": s.id, "scope": scope, "sample": s.meta})
        if task == "tool_select":
            enc.tool_label = 0
        else:
            enc.kb_soft = [0.0] * len(enc.option_ids)
        return enc

    if task == "role_select":
        text = _render_role_scope(sample, scope)
        tok = _tokenize(tokenizer, text, max_len)
        if tok is None:
            return None
        ids, _ = tok
        return Encoded(task=task, record_id=f"{sample.id}::{scope}", role=None, input_ids=ids,
                       prompt_len=len(ids), q_position=len(ids) - 1, class_label=0,
                       meta={"text": text, "base_id": sample.id, "scope": scope, "sample": sample.meta})
    if task == "relay_select":
        text = _render_relay_scope(sample, scope)
        tok = _tokenize(tokenizer, text, max_len)
        if tok is None:
            return None
        ids, _ = tok
        return Encoded(task=task, record_id=f"{sample.id}::{scope}", role=None, input_ids=ids,
                       prompt_len=len(ids), q_position=len(ids) - 1, class_label=0,
                       meta={"text": text, "base_id": sample.id, "scope": scope, "sample": sample.meta})
    if task == "context_retention":
        text, spans = _render_context_scope(sample, scope)
        tok = _tokenize(tokenizer, text, max_len)
        if tok is None:
            return None
        ids, offsets = tok
        positions = char_spans_to_token_indices(offsets, spans)
        if len(positions) != len(sample.groups):
            raise ValueError(f"{sample.id}: group span mismatch")
        return Encoded(task=task, record_id=f"{sample.id}::{scope}", role=None, input_ids=ids,
                       prompt_len=len(ids), q_position=len(ids) - 1,
                       group_positions=positions, group_labels=[0] * len(positions),
                       meta={"text": text, "base_id": sample.id, "scope": scope,
                             "sample": sample.meta, "group_seqs": [int(g["seq"]) for g in sample.groups]})

    # premise / overclaim: context field is a separate <context> block
    s = scope_sample(sample, task, scope)
    from ..collate import encode_classification
    enc = encode_classification(s, tokenizer, max_len)
    if enc is None:
        return None
    enc.record_id = f"{s.id}::{scope}"
    enc.meta = {**enc.meta, "base_id": s.id, "scope": scope}
    return enc

def load_scope_bases(data_dir: str, tasks: Sequence[str] = TASKS, split: str = "test") -> dict[str, list[Sample]]:
    catalog = load_catalog_snapshot()
    out = {}
    for task in tasks:
        out[task] = load_split(data_dir, task, split, catalog)
    return out


def predict_scope(
    model,
    tokenizer,
    samples_by_task: dict[str, list[Sample]],
    scopes: Sequence[str] = ("local", "global", "local_len", "global_len"),
    cfg: dict | None = None,
) -> tuple[dict, list[dict]]:
    catalog = load_catalog_snapshot()
    metrics: dict = {}
    rows: list[dict] = []
    for task, samples in samples_by_task.items():
        for scope in scopes:
            encoded = []
            skipped = 0
            for s in samples:
                enc = encode_scope(s, tokenizer, task, scope, catalog)
                if enc is None:
                    skipped += 1
                else:
                    encoded.append(enc)
            preds = predict_encoded(model, tokenizer, encoded, task, cfg)
            by_vid = {p["id"]: p for p in preds}
            preds_orig: list[dict] = []
            for s in samples:
                p = by_vid.get(f"{s.id}::{scope}")
                if p is None:
                    continue
                q = dict(p)
                q["id"] = s.id
                preds_orig.append(q)
                rows.append({
                    "task": task, "base_id": s.id, "scope": scope,
                    "skipped": skipped, "pred": p,
                })
            metrics[f"{task}::{scope}"] = compute(task, samples, preds_orig)
    return metrics, rows
