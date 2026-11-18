#!/usr/bin/env python3
"""Render Paper 1 result figures from the JSON summaries."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="artifacts/eval/paper1/scope2/summary.json")
    ap.add_argument("--constraint", default="artifacts/eval/paper1/constraint/summary.json")
    ap.add_argument("--out", default="paperwork1/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.scope, encoding="utf-8") as fh:
        scope = json.load(fh)
    with open(args.constraint, encoding="utf-8") as fh:
        con = json.load(fh)

    tasks = ["tool", "kb", "context", "role", "relay", "premise", "flag"]
    labels = {"tool": "tool", "kb": "kb", "context": "context", "role": "role",
              "relay": "relay", "premise": "premise", "flag": "flag"}
    keys = {"tool": "tool_select", "kb": "kb_select", "context": "context_retention",
            "role": "role_select", "relay": "relay_select", "premise": "premise_judge",
            "flag": "overclaim_flag"}

    # E1: local vs global, with local_len control
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(tasks))
    loc = [scope[keys[t]]["local"] for t in tasks]
    glob = [scope[keys[t]]["global"] for t in tasks]
    ax.bar(x - 0.18, loc, 0.36, label="local", color="#4C72B0")
    ax.bar(x + 0.18, glob, 0.36, label="global", color="#C44E52")
    ax.set_xticks(x, [labels[t] for t in tasks], rotation=20)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("task score")
    ax.set_title("E1: local vs global observation")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_e1_scope.png"), dpi=200)
    fig.savefig(os.path.join(args.out, "fig_e1_scope.pdf"))

    # E2: four modes
    e2_tasks = ["tool_select", "kb_select", "role_select", "relay_select"]
    modes = ["open", "authorized", "postfilter", "random_size"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(e2_tasks))
    width = 0.2
    colors = {"open": "#4C72B0", "authorized": "#C44E52", "postfilter": "#55A868", "random_size": "#8172B2"}
    for i, m in enumerate(modes):
        vals = []
        for t in e2_tasks:
            ms = con[t]["per_model"]
            vals.append(np.mean([ms[k][m]["exec"] for k in ms if m in ms[k]]))
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=colors[m])
    ax.set_xticks(x, [t.replace("_select", "") for t in e2_tasks])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("executable accuracy")
    ax.set_title("E2: candidate constraint placement")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_e2_constraint.png"), dpi=200)
    fig.savefig(os.path.join(args.out, "fig_e2_constraint.pdf"))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
