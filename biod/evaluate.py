"""Evaluation harness for the zero-training baseline (and later, trained heads).

The eval set is JSONL, one item per line::

    {"id":"t01","kind":"tool","expected_domain":"生信通用流程与工具链",
     "expected":"bio-enrich","state":"我有一批差异表达基因列表……"}
    {"id":"k01","kind":"kb","expected":["科学通用/假设检验与p值"],"state":"……"}

Metrics
-------
tool : entry Top-1 accuracy, domain Top-1 accuracy, latency
kb   : hit rate (expected ⊂ selected), average selected count, empty rate
"""

from __future__ import annotations

import json
import statistics
from typing import Sequence

from .decide import DecisionEngine


def load_eval_set(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def evaluate(
    engine: DecisionEngine,
    items: Sequence[dict],
    mode: str = "auto",
    tau: float = 0.5,
    top_k: int = 5,
    records_path: str | None = None,
    tool_instruction: str | None = None,
    kb_instruction: str | None = None,
    tool_mode: str | None = None,
    kb_mode: str | None = None,
    tool_pointwise_instruction: str | None = None,
) -> dict:
    rows = []
    replay_records: list[dict] = []
    latencies = []
    for item in items:
        kind = item["kind"]
        expected = item.get("expected")
        if kind == "tool":
            kwargs = {}
            if tool_instruction:
                kwargs["instruction"] = tool_instruction
            if tool_pointwise_instruction:
                kwargs["pointwise_instruction"] = tool_pointwise_instruction
            dec = engine.decide_tool(item["state"], mode=tool_mode or mode, **kwargs)
            pred = dec.selected[0] if dec.selected else None
            pred_domain = dec.steps[0].chosen[0] if dec.steps and dec.steps[0].chosen else None
            rows.append(
                {
                    "id": item.get("id"),
                    "kind": kind,
                    "expected": expected,
                    "expectedDomain": item.get("expected_domain"),
                    "predicted": pred,
                    "predictedDomain": pred_domain,
                    "hit": pred == expected,
                    "domainHit": pred_domain == item.get("expected_domain"),
                    "latencyMs": round(dec.latency_ms, 1),
                }
            )
        else:
            expected_ids = expected if isinstance(expected, list) else ([expected] if expected else [])
            kwargs = {"instruction": kb_instruction} if kb_instruction else {}
            dec = engine.decide_kb(item["state"], mode=kb_mode or mode, tau=tau, max_selected=top_k, **kwargs)
            pred = list(dec.selected)
            hit = bool(set(expected_ids) & set(pred)) if expected_ids else (len(pred) == 0)
            rows.append(
                {
                    "id": item.get("id"),
                    "kind": kind,
                    "expected": expected_ids,
                    "predicted": pred,
                    "hit": hit,
                    "selectedCount": len(pred),
                    "latencyMs": round(dec.latency_ms, 1),
                }
            )
        latencies.append(rows[-1]["latencyMs"])
        # keep the replayable steps (+ labels) so Dream-RSI style replay can
        # sweep selection policies offline at zero model cost
        if records_path:
            expected_ids = expected if isinstance(expected, list) else ([expected] if expected else [])
            for rec in dec.records():
                rec["source"] = "eval"
                rec["evalId"] = item.get("id")
                if kind == "tool":
                    rec["label"] = expected
                else:
                    rec["labels"] = list(expected_ids)
                replay_records.append(rec)

    tool_rows = [r for r in rows if r["kind"] == "tool"]
    kb_rows = [r for r in rows if r["kind"] == "kb"]
    summary = {
        "items": len(rows),
        "tool_items": len(tool_rows),
        "tool_entry_accuracy": round(sum(r["hit"] for r in tool_rows) / len(tool_rows), 4) if tool_rows else None,
        "tool_domain_accuracy": round(sum(r["domainHit"] for r in tool_rows) / len(tool_rows), 4) if tool_rows else None,
        "kb_items": len(kb_rows),
        "kb_hit_rate": round(sum(r["hit"] for r in kb_rows) / len(kb_rows), 4) if kb_rows else None,
        "kb_avg_selected": round(statistics.mean(r["selectedCount"] for r in kb_rows), 2) if kb_rows else None,
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "mode": mode,
        "toolMode": tool_mode or mode,
        "kbMode": kb_mode or mode,
        "tau": tau,
        "toolInstruction": tool_instruction,
        "kbInstruction": kb_instruction,
        "toolPointwiseInstruction": tool_pointwise_instruction,
        "noToolMargin": getattr(engine, "no_tool_margin", 0.0),
        "backend": engine.backend.name,
    }
    if records_path:
        import json as _json
        with open(records_path, "w", encoding="utf-8") as fh:
            for rec in replay_records:
                fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    return {"summary": summary, "rows": rows, "recordsPath": records_path}


def format_table(result: dict) -> str:
    lines = []
    s = result["summary"]
    lines.append(f"backend={s['backend']} mode={s['mode']} tau={s['tau']}")
    lines.append(
        f"tool: entry_acc={s['tool_entry_accuracy']} domain_acc={s['tool_domain_accuracy']} "
        f"(n={s['tool_items']})"
    )
    lines.append(f"kb:   hit_rate={s['kb_hit_rate']} avg_selected={s['kb_avg_selected']} (n={s['kb_items']})")
    lines.append(f"latency p50 = {s['latency_p50_ms']} ms")
    lines.append("")
    lines.append(f"{'id':<5}{'kind':<6}{'hit':<6}expected / predicted")
    for r in result["rows"]:
        exp = r["expected"] if isinstance(r["expected"], str) else ",".join(r["expected"])
        pred = r["predicted"] if isinstance(r["predicted"], str) else ",".join(r["predicted"] or [])
        lines.append(f"{r['id']:<5}{r['kind']:<6}{str(r['hit']):<6}{exp}  ->  {pred}")
    return "\n".join(lines)
