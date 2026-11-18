#!/usr/bin/env python3
"""Summarise E1 scope probes (local/global/length) with paired item bootstrap."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from biod.multitask.catalog import load_catalog_snapshot
from biod.multitask.data import load_split

TASKS = ("tool_select", "kb_select", "context_retention", "role_select", "relay_select", "premise_judge", "overclaim_flag")
SCOPES = ("local", "global", "local_len", "global_len")


def ok(task, s, p):
    if task == "tool_select":
        return 1.0 if p.get("chosen_id") == (s.label or {}).get("id") else 0.0
    if task == "kb_select":
        grades = (s.label or {}).get("grades") or {}
        g = {k for k, v in grades.items() if v == 2}
        if not g:
            return None
        scores = p.get("scores") or {}
        top = max(scores, key=scores.get) if scores else None
        return 1.0 if top in g else 0.0
    if task == "context_retention":
        pm = {x["seq"]: x["label"] for x in p.get("labels", [])}
        xs = [1.0 if pm.get(x["seq"]) == x["label"] else 0.0 for x in s.label if x["seq"] in pm]
        return statistics.mean(xs) if xs else None
    if task == "role_select":
        return 1.0 if p.get("role") == (s.label or {}).get("role") else 0.0
    if task == "relay_select":
        return 1.0 if p.get("verdict") == (s.label or {}).get("verdict") else 0.0
    if task == "premise_judge":
        return 1.0 if p.get("relation") == (s.label or {}).get("relation") else 0.0
    if task == "overclaim_flag":
        return 1.0 if set(p.get("flags") or []) == set((s.label or {}).get("flags") or []) else 0.0
    raise ValueError(task)


def boot_ci(values, n=2000, seed=20261104):
    rng = random.Random(seed)
    vals = list(values)
    out = []
    for _ in range(n):
        out.append(statistics.mean(rng.choice(vals) for _ in vals))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope-dir", default="artifacts/eval/paper1/scope")
    ap.add_argument("--data-dir", default="Datasets/paper1/legacy-test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    catalog = load_catalog_snapshot()
    gold = {t: {s.id: s for s in load_split(args.data_dir, t, "test", catalog)} for t in TASKS}
    model_files = sorted(glob.glob(os.path.join(args.scope_dir, "paper1-*.json")))
    # vals[(task, model, scope)][base] = score
    vals = defaultdict(dict)
    model_cap = {}
    for path in model_files:
        name = os.path.basename(path).replace(".json", "")
        m = re.match(r"paper1-(2B|4B|9B)-s(\d+)", name)
        if not m:
            continue
        cap = m.group(1)
        model_cap[name] = cap
        pred_path = os.path.splitext(path)[0] + "_predictions.jsonl"
        if not os.path.isfile(pred_path):
            continue
        by = defaultdict(dict)
        for line in open(pred_path, encoding="utf-8"):
            r = json.loads(line)
            by[(r["task"], r["scope"])][r["base_id"]] = r["pred"]
        for task in TASKS:
            for scope in SCOPES:
                for sid, p in by.get((task, scope), {}).items():
                    s = gold[task].get(sid)
                    if s is None:
                        continue
                    v = ok(task, s, p)
                    if v is not None:
                        vals[(task, name, scope)][sid] = v

    summary = {}
    for task in TASKS:
        models = sorted({k[1] for k in vals if k[0] == task})
        if not models:
            continue
        # per-model means
        per_model = {}
        for model in models:
            per_model[model] = {sc: round(statistics.mean(vals[(task, model, sc)].values()), 6)
                                for sc in SCOPES if vals[(task, model, sc)]}
        # paired global-local averaged over models, by base
        base_ids = sorted(set.intersection(*[set(vals[(task, m, "local")]) for m in models],
                                           *[set(vals[(task, m, "global")]) for m in models]))
        deltas = []
        for sid in base_ids:
            l = statistics.mean(vals[(task, m, "local")][sid] for m in models)
            g = statistics.mean(vals[(task, m, "global")][sid] for m in models)
            deltas.append(g - l)
        d = statistics.mean(deltas) if deltas else 0.0
        ci = boot_ci(deltas) if deltas else (0.0, 0.0)
        # length controls
        def contrast(scope):
            ids = sorted(set.intersection(*[set(vals[(task, m, "local")]) for m in models],
                                           *[set(vals[(task, m, scope)]) for m in models]))
            ds = []
            for sid in ids:
                l = statistics.mean(vals[(task, m, "local")][sid] for m in models)
                x = statistics.mean(vals[(task, m, scope)][sid] for m in models)
                ds.append(x - l)
            return statistics.mean(ds) if ds else 0.0, boot_ci(ds) if ds else (0.0, 0.0), len(ds)
        ll = contrast("local_len")
        gl = contrast("global_len")
        by_cap = {}
        for cap in ("2B", "4B", "9B"):
            ms = [m for m in models if model_cap.get(m) == cap]
            if not ms:
                continue
            ds = []
            for sid in base_ids:
                l = statistics.mean(vals[(task, m, "local")][sid] for m in ms)
                g = statistics.mean(vals[(task, m, "global")][sid] for m in ms)
                ds.append(g - l)
            by_cap[cap] = {"delta": round(statistics.mean(ds), 6),
                           "ci": [round(x, 6) for x in boot_ci(ds)],
                           "n_base": len(ds), "n_models": len(ms)}
        summary[task] = {
            "models": models,
            "n_base": len(base_ids),
            "local": round(statistics.mean(per_model[m]["local"] for m in models), 6),
            "global": round(statistics.mean(per_model[m]["global"] for m in models), 6),
            "delta_global_minus_local": round(d, 6),
            "ci": [round(x, 6) for x in ci],
            "local_len_minus_local": {"delta": round(ll[0], 6), "ci": [round(x, 6) for x in ll[1]], "n_base": ll[2]},
            "global_len_minus_local": {"delta": round(gl[0], 6), "ci": [round(x, 6) for x in gl[1]], "n_base": gl[2]},
            "by_capacity": by_cap,
            "per_model": per_model,
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    for task, s in summary.items():
        print(task, "n", s["n_base"], "local", s["local"], "global", s["global"],
              "delta", s["delta_global_minus_local"], "ci", s["ci"],
              "cap", {k: v["delta"] for k, v in s["by_capacity"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
