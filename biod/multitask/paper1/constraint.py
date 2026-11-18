"""Paper 1 E2 candidate-constraint manipulation on frozen test items.

Modes:
  open            full role-visible candidate set
  authorized      policy subset (input-domain restriction)
  postfilter      full set scored, disallowed choice mapped to fallback
  random_size     random subset with the same size as authorized (size control)

The label is fixed; only the action space / decode rule changes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from transformers import AutoTokenizer

from biod.render import render_kb_prompt, render_tool_prompt

from ..catalog import load_catalog_snapshot
from ..collate import Encoded, _tokenize, char_spans_to_token_indices, encode_classification
from ..data import Sample, load_split
from ..infer import predict_encoded
from ..tasks import TASK_SPECS, TASKS

ROLES = ("pi", "searcher", "researcher", "writer", "reviewer")
VERDICTS = ("allow", "reject", "escalate")
TASKS_E2 = ("tool_select", "kb_select", "role_select", "relay_select")
MODES = ("open", "authorized", "postfilter", "random_size")


@dataclass
class Ctx:
    mode: str
    allowed: tuple[str, ...]
    options: tuple  # CatalogOption list for pointer, () for classification
    fallback: str
    ideal: object
    coverage: bool


def _rng(base_id: str, mode: str) -> random.Random:
    h = int(hashlib.sha256(f"{base_id}|{mode}".encode()).hexdigest()[:16], 16)
    return random.Random(h)


def _split_ids(all_ids: Sequence[str], ideal: Sequence[str], coverage: bool, rng: random.Random) -> list[str]:
    ideal_set = set(ideal)
    excluded_n = max(1, int(round(0.30 * len(all_ids))))
    candidates = [x for x in all_ids if x not in ideal_set]
    rng.shuffle(candidates)
    if coverage:
        out = [x for x in all_ids if x not in ideal_set]
        # remove ~30% including all ideal
        keep = [x for x in out if x not in set(candidates[:excluded_n])]
        return keep
    # keep ideal, remove ~30% from non-ideal
    keep = ideal_set | set(candidates[excluded_n:])
    return [x for x in all_ids if x in keep]


def _authorized(sample: Sample, task: str, coverage: bool) -> tuple[list[str], str]:
    if task == "tool_select":
        all_ids = [o.id for o in sample.options] + ["NO_TOOL"]
        ideal = [(sample.label or {}).get("id")]
        allowed = _split_ids(all_ids, ideal, coverage, _rng(sample.id, "authorized"))
        # NO_TOOL is always allowed
        if "NO_TOOL" not in allowed:
            allowed.append("NO_TOOL")
        return allowed, "NO_TOOL"
    if task == "kb_select":
        all_ids = [o.id for o in sample.options]
        grades = (sample.label or {}).get("grades") or {}
        ideal = [k for k, v in grades.items() if v == 2]
        return _split_ids(all_ids, ideal, coverage, _rng(sample.id, "authorized")), ""
    if task == "role_select":
        ideal = (sample.label or {})["role"]
        allowed = _split_ids(list(ROLES), [ideal], coverage, _rng(sample.id, "authorized"))
        return allowed, "pi"
    if task == "relay_select":
        ideal = (sample.label or {})["verdict"]
        allowed = _split_ids(list(VERDICTS), [ideal], coverage, _rng(sample.id, "authorized"))
        return allowed, "escalate"
    raise ValueError(task)


def _random_size(sample: Sample, task: str, authorized: Sequence[str], fallback: str) -> list[str]:
    if task in {"tool_select", "kb_select"}:
        all_ids = [o.id for o in sample.options] + (["NO_TOOL"] if task == "tool_select" else [])
        n = len([x for x in authorized if x != "NO_TOOL"])
        rng = _rng(sample.id, "random")
        pool = [x for x in all_ids if x != "NO_TOOL"]
        rng.shuffle(pool)
        out = pool[:n]
        if task == "tool_select" and "NO_TOOL" not in out:
            out.append("NO_TOOL")
        return out
    return _split_ids(list(ROLES if task == "role_select" else VERDICTS),
                      [(sample.label or {})["role" if task == "role_select" else "verdict"]],
                      False, _rng(sample.id, "random"))


def build_ctx(sample: Sample, task: str, mode: str) -> Ctx:
    if mode == "open":
        if task == "tool_select":
            return Ctx(mode, tuple([o.id for o in sample.options] + ["NO_TOOL"]), tuple(sample.options), "NO_TOOL", (sample.label or {}).get("id"), False)
        if task == "kb_select":
            grades = (sample.label or {}).get("grades") or {}
            ideal = [k for k, v in grades.items() if v == 2]
            return Ctx(mode, tuple(o.id for o in sample.options), tuple(sample.options), "", ideal, False)
        if task == "role_select":
            return Ctx(mode, ROLES, (), "pi", (sample.label or {})["role"], False)
        return Ctx(mode, VERDICTS, (), "escalate", (sample.label or {})["verdict"], False)

    if task == "tool_select" or task == "kb_select":
        full = [o.id for o in sample.options] + (["NO_TOOL"] if task == "tool_select" else [])
        ideal = (sample.label or {}).get("id") if task == "tool_select" else [k for k, v in ((sample.label or {}).get("grades") or {}).items() if v == 2]
        ideal_seq = ideal if isinstance(ideal, list) else [ideal]
        # deterministic coverage: 50% of bases lose the ideal
        coverage = int(hashlib.sha256(f"{sample.id}|cov".encode()).hexdigest(), 16) % 2 == 0
        allowed, fallback = _authorized(sample, task, coverage)
        if mode == "authorized" or mode == "postfilter":
            pass
        elif mode == "random_size":
            allowed = _random_size(sample, task, allowed, fallback)
            coverage = not all(x in allowed for x in ideal_seq)
        else:
            raise ValueError(mode)
        options = [o for o in sample.options if o.id in set(allowed)]
        # postfilter prompts/options use full set; allowed still policy set
        prompt_options = sample.options if mode == "postfilter" else options
        return Ctx(mode, tuple(allowed), tuple(prompt_options), fallback, ideal, coverage)

    # role/relay
    ideal = (sample.label or {})["role" if task == "role_select" else "verdict"]
    coverage = int(hashlib.sha256(f"{sample.id}|cov".encode()).hexdigest(), 16) % 2 == 0
    allowed, fallback = _authorized(sample, task, coverage)
    if mode == "random_size":
        allowed = _random_size(sample, task, allowed, fallback)
        coverage = ideal not in allowed
    return Ctx(mode, tuple(allowed), (), fallback, ideal, coverage)


def encode_ctx(sample: Sample, tokenizer, task: str, ctx: Ctx) -> Encoded | None:
    max_len = TASK_SPECS[task].max_len
    if task in {"tool_select", "kb_select"}:
        state = sample.state_text
        if sample.role:
            state = f"[角色：{sample.role}]\n{state}"
        prompt = render_tool_prompt(state, list(ctx.options)) if task == "tool_select" else render_kb_prompt(state, list(ctx.options))
        tok = _tokenize(tokenizer, prompt.text, max_len)
        if tok is None:
            return None
        ids, offsets = tok
        positions = char_spans_to_token_indices(offsets, prompt.option_spans)
        enc = Encoded(task=task, record_id=f"{sample.id}::{ctx.mode}", role=sample.role, input_ids=ids,
                      prompt_len=len(ids), q_position=len(ids) - 1, option_positions=positions,
                      option_ids=list(prompt.option_ids),
                      meta={"base_id": sample.id, "mode": ctx.mode, "allowed": list(ctx.allowed)})
        if task == "tool_select":
            enc.tool_label = 0
        else:
            enc.kb_soft = [0.0] * len(enc.option_ids)
        return enc
    enc = encode_classification(sample, tokenizer, max_len)
    if enc is None:
        return None
    enc.record_id = f"{sample.id}::{ctx.mode}"
    enc.meta = {**enc.meta, "base_id": sample.id, "mode": ctx.mode, "allowed": list(ctx.allowed)}
    return enc


def _tool_pred(p: dict) -> str:
    return str(p.get("chosen_id"))


def _kb_pred(p: dict) -> list[str]:
    return list(p.get("selected_ids") or [])


def _class_pred(p: dict, task: str) -> str:
    if task == "role_select":
        return str(p.get("role"))
    return str(p.get("verdict"))


def _apply(ctx: Ctx, task: str, p: dict) -> dict:
    """Apply postfilter and score strict/executable correctness."""
    if task == "tool_select":
        raw = _tool_pred(p)
        chosen = raw
        if ctx.mode == "postfilter" and raw not in ctx.allowed:
            chosen = ctx.fallback
        strict = chosen == ctx.ideal
        exec_ok = (chosen == ctx.ideal) if not ctx.coverage else (chosen == ctx.fallback)
        return {"raw": raw, "chosen": chosen, "strict": bool(strict), "exec": bool(exec_ok), "fallback": chosen == ctx.fallback and raw != ctx.fallback}
    if task == "kb_select":
        raw = _kb_pred(p)
        if ctx.mode == "postfilter":
            chosen = [x for x in raw if x in ctx.allowed]
        else:
            chosen = list(raw)
        allowed_ideal = [x for x in (ctx.ideal or []) if x in ctx.allowed]
        if allowed_ideal:
            scores = p.get("scores") or {}
            top = max(scores, key=scores.get) if scores else None
            exec_ok = top in allowed_ideal
        else:
            exec_ok = len(chosen) == 0
        strict = bool(ctx.ideal) and set(ctx.ideal) <= set(chosen)
        return {"raw": raw, "chosen": chosen, "strict": bool(strict), "exec": bool(exec_ok), "fallback": chosen == [] and len(raw) > 0}
    # role / relay
    raw = _class_pred(p, task)
    if ctx.mode in {"authorized", "random_size"}:
        probs = p.get("probs") or {}
        pool = [x for x in ctx.allowed if x in probs]
        chosen = max(pool, key=probs.get) if pool else ctx.fallback
    else:
        chosen = raw
        if ctx.mode == "postfilter" and raw not in ctx.allowed:
            chosen = ctx.fallback
    strict = chosen == ctx.ideal
    exec_ok = (chosen == ctx.ideal) if not ctx.coverage else (chosen == ctx.fallback)
    return {"raw": raw, "chosen": chosen, "strict": bool(strict), "exec": bool(exec_ok), "fallback": chosen == ctx.fallback and raw != ctx.fallback}


def evaluate_constraints(model, tokenizer, samples_by_task: dict[str, list[Sample]], modes: Sequence[str] = MODES, cfg: dict | None = None) -> tuple[dict, list[dict]]:
    metrics: dict = {}
    rows: list[dict] = []
    for task, samples in samples_by_task.items():
        if task not in TASKS_E2:
            continue
        for mode in modes:
            encoded = []
            ctxs = {}
            for s in samples:
                ctx = build_ctx(s, task, mode)
                enc = encode_ctx(s, tokenizer, task, ctx)
                if enc is None:
                    continue
                ctxs[f"{s.id}::{mode}"] = ctx
                encoded.append(enc)
            preds = predict_encoded(model, tokenizer, encoded, task, cfg)
            n = 0
            n_exec = n_strict = n_cov = n_fb = 0
            for p in preds:
                vid = p["id"]
                ctx = ctxs[vid]
                a = _apply(ctx, task, p)
                n += 1
                n_exec += int(a["exec"])
                n_strict += int(a["strict"])
                n_cov += int(ctx.coverage)
                n_fb += int(a["fallback"])
                rows.append({"task": task, "base_id": ctx.meta_id if hasattr(ctx, "meta_id") else vid.split("::")[0],
                             "mode": mode, "coverage": ctx.coverage, "ideal": ctx.ideal,
                             "allowed": list(ctx.allowed), **a})
            metrics[f"{task}::{mode}"] = {
                "n": n, "exec": n_exec / max(1, n), "strict": n_strict / max(1, n),
                "coverage_rate": n_cov / max(1, n), "fallback_rate": n_fb / max(1, n),
            }
    return metrics, rows


def load_bases(data_dir: str, tasks: Sequence[str] = TASKS_E2, split: str = "test") -> dict[str, list[Sample]]:
    catalog = load_catalog_snapshot()
    return {t: load_split(data_dir, t, split, catalog) for t in tasks}
