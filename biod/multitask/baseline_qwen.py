"""Qwen3.8-Flash-Next baseline for the five decisions.

Same state and candidate set as the student, but the baseline is generative, so
the prompt includes candidate ids and asks for JSON. Used for the head-to-head
evaluation; not part of training.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Sequence

from .catalog import CatalogOption
from .data import Sample
from .tasks import CONTEXT_LABELS, RELAY_VERDICTS, ROLES

DEFAULT_BASE_URL = "https://your-llm-endpoint/v1"
DEFAULT_MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"


def _env_key() -> str:
    key = os.environ.get("BIOFLOW_LLM_API_KEY_DGX", "")
    if key:
        return key
    env_path = "/path/to/bioflow/agent-client/backend/.env"
    if os.path.isfile(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("BIOFLOW_LLM_API_KEY_DGX="):
                return line.split("=", 1)[1]
    return ""


def _option_line(i: int, opt: CatalogOption, desc_chars: int = 160) -> str:
    desc = opt.description.strip()
    if len(desc) > desc_chars:
        desc = desc[:desc_chars].rstrip() + "…"
    return f"{i}. {opt.name}（{opt.kind}：{opt.id}）— {desc}"


def render_qwen(sample: Sample) -> str:
    task = sample.task
    if task == "tool_select":
        opts = "\n".join(_option_line(i + 1, o) for i, o in enumerate(sample.options))
        return f"""<task>tool_select</task>
<role>{sample.role}</role>
<state>
{sample.state_text[:800]}
</state>
<options>
{opts}
0. NO_TOOL — 以上都不合适
</options>
从下列能力中选一个下一步最该加载的；没有适用项选 NO_TOOL。只输出 JSON：{{"id":"<候选 id 或 NO_TOOL>"}}，不要解释。"""
    if task == "kb_select":
        opts = "\n".join(_option_line(i + 1, o) for i, o in enumerate(sample.options))
        return f"""<task>kb_select</task>
<role>{sample.role}</role>
<state>
{sample.state_text[:1600]}
</state>
<candidates>
{opts}
</candidates>
判断每条知识对当前这一步是否有用；可以多选，也可以一条都不选。只输出 JSON：{{"selected":["<候选 id>", "..."]}}，不要解释。"""
    if task == "context_retention":
        blocks = []
        for g in sample.groups:
            blocks.append(f"[seq {g['seq']}] {g.get('kind','tool')} / {g.get('tool','') or 'n/a'} / {g.get('outcome','unknown')}\n{g.get('text','')}")
        return f"""<task>context_retention</task>
<state>
{sample.state_text[:1000]}
</state>
<groups>
{chr(10).join(blocks)}
</groups>
对每组历史：原文保留（keep）/ 摘要（summarise）/ 丢弃（drop）。只输出 JSON 数组：[{{"seq":12,"label":"keep"}}, ...]，不要解释。"""
    if task == "role_select":
        roster = "\n".join(f"- {r}" for r in ROLES)
        return f"""<task>role_select</task>
<team>
{roster}
</team>
<policy>单个非 PI 角色能独立完成就选它；需要跨角色调度、验收/仲裁、全流程或不确定时选 pi。</policy>
<state>
{sample.state_text[:2000]}
</state>
只输出 JSON：{{"role":"pi|searcher|researcher|writer|reviewer","confidence":0.0}}，不要解释。"""
    if task == "relay_select":
        return f"""<task>relay_select</task>
<from>{sample.from_role}</from>
<to>{sample.to_role}</to>
<why>{sample.why[:400]}</why>
<body>
{sample.body[:1500]}
</body>
<policy>allow = 必要、目标负责、无越权；reject = 不必要/目标不负责/重复劳动/越权；escalate = 模糊、高风险、需要 PI 仲裁或不确定。</policy>
只输出 JSON：{{"verdict":"allow|reject|escalate","confidence":0.0}}，不要解释。"""
    raise ValueError(f"unknown task {task!r}")


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def call_qwen(prompt: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
              max_tokens: int = 400, timeout: int = 120) -> dict:
    key = _env_key()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Decision baselines should be System-1: without this Qwen emits
        # reasoning_content and leaves content empty (observed on the dgx
        # endpoint), which is both slow and unparseable.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    latency = time.time() - started
    message = payload.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content") or ""
    return {"content": content, "json": _extract_json(content), "latency_ms": latency * 1000.0}


def decode_baseline(sample: Sample, payload: dict) -> dict:
    task = sample.task
    data = payload.get("json")
    ids = [o.id for o in sample.options]
    if task == "tool_select":
        raw = str((data or {}).get("id", "")).strip()
        if raw.isdigit():
            slot = int(raw)
            raw = "NO_TOOL" if slot == 0 else (ids[slot - 1] if 1 <= slot <= len(ids) else "")
        if raw not in ids and raw != "NO_TOOL":
            raw = "NO_TOOL"
        ranked = ([raw] if raw in ids else []) + [i for i in ids if i != raw]
        return {"chosen_id": raw, "ranked_ids": ranked, "confidence": 0.5,
                "latency_ms": payload["latency_ms"], "raw": payload["content"][:500]}
    if task == "kb_select":
        selected = (data or {}).get("selected", []) if isinstance(data, dict) else (data or [])
        cleaned = []
        by_name = {o.name: o.id for o in sample.options}
        for item in selected if isinstance(selected, list) else []:
            raw = str(item).strip()
            if raw.isdigit() and 1 <= int(raw) <= len(ids):
                cleaned.append(ids[int(raw) - 1])
            elif raw in ids:
                cleaned.append(raw)
            elif raw in by_name:
                cleaned.append(by_name[raw])
        scores = {i: (0.9 if i in cleaned else 0.1) for i in ids}
        return {"selected_ids": cleaned[:3], "scores": scores, "latency_ms": payload["latency_ms"], "raw": payload["content"][:500]}
    if task == "context_retention":
        labels = []
        if isinstance(data, dict):
            data = [data]
        by_seq = {}
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and "seq" in item:
                by_seq[int(item["seq"])] = str(item.get("label", "summarise"))
        for g in sample.groups:
            labels.append({"seq": g["seq"], "label": by_seq.get(g["seq"], "summarise"), "confidence": 0.5})
        return {"labels": labels, "latency_ms": payload["latency_ms"], "raw": payload["content"][:500]}
    if task == "role_select":
        role = str((data or {}).get("role", "")) if isinstance(data, dict) else ""
        if role not in ROLES:
            role = "pi"
        return {"role": role, "confidence": 0.5, "latency_ms": payload["latency_ms"], "raw": payload["content"][:500]}
    if task == "relay_select":
        verdict = str((data or {}).get("verdict", "")) if isinstance(data, dict) else ""
        if verdict not in RELAY_VERDICTS:
            verdict = "escalate"
        return {"verdict": verdict, "confidence": 0.5, "latency_ms": payload["latency_ms"], "raw": payload["content"][:500]}
    raise ValueError(task)


def predict_sample(sample: Sample, retries: int = 2) -> dict:
    prompt = render_qwen(sample)
    last = None
    for _ in range(retries + 1):
        try:
            payload = call_qwen(prompt)
            if payload["json"] is not None:
                return decode_baseline(sample, payload)
            last = payload
        except Exception as exc:  # network/parse; retry
            last = {"content": f"{type(exc).__name__}: {exc}", "json": None, "latency_ms": 0.0}
    return decode_baseline(sample, last or {"content": "", "json": None, "latency_ms": 0.0})
