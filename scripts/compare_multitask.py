#!/usr/bin/env python3
"""Compare student and Qwen baseline metrics on the same frozen test split."""
from __future__ import annotations
import argparse, json

KEYS = {
    "tool_select": [("top1_strict", "Top-1 strict"), ("no_tool_f1", "NO_TOOL F1"), ("macro_top1", "Macro Top-1"), ("ece", "ECE")],
    "kb_select": [("hit@1", "Hit@1"), ("r@5", "R@5"), ("empty_f1", "Empty F1")],
    "context_retention": [("accuracy_3", "3-class acc"), ("macro_f1", "Macro F1")],
    "role_select": [("accuracy_5", "5-class acc"), ("macro_f1", "Macro F1"), ("pi_recall_cross_role", "PI recall (cross-role)"), ("misroute_rate", "Misroute rate")],
    "relay_select": [("accuracy_3", "3-class acc"), ("macro_f1", "Macro F1"), ("unsafe_allow_rate", "Unsafe-allow"), ("escalate_recall", "Escalate recall")],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.load(open(args.student, encoding="utf-8"))["metrics"]
    q = json.load(open(args.qwen, encoding="utf-8"))["metrics"]
    lines = ["# Student vs Qwen3.8-Flash-Next (frozen test split)", "", "| Task | Metric | Student | Qwen | Δ (pp) | Verdict |", "|---|---|---:|---:|---:|---|"]
    for task, keys in KEYS.items():
        if task not in s or task not in q:
            continue
        for key, label in keys:
            sv = s[task].get(key)
            qv = q[task].get(key)
            if sv is None or qv is None:
                continue
            delta = (float(sv) - float(qv)) * 100
            # lower is better for error/rate metrics
            lower_better = key in {"ece", "misroute_rate", "unsafe_allow_rate"}
            ok = abs(delta) <= 2.0
            verdict = "✅" if ok else ("✅ better" if (delta < 0 if lower_better else delta > 0) else "⚠️")
            lines.append(f"| {task} | {label} | {float(sv):.4f} | {float(qv):.4f} | {delta:+.1f} | {verdict} |")
    lines += ["", "Δ = student − qwen, 单位 pp；误差/率类指标越低越好。", ""]
    text = "\n".join(lines)
    open(args.out, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
