#!/usr/bin/env python3
"""Summarise E2 candidate-constraint probes with paired item bootstrap."""
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

MODES = ("open", "authorized", "postfilter", "random_size")
TASKS = ("tool_select", "kb_select", "role_select", "relay_select")


def boot_ci(xs, n=2000, seed=20261104):
    rng = random.Random(seed)
    xs = list(xs)
    out = []
    for _ in range(n):
        out.append(statistics.mean(rng.choice(xs) for _ in xs))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/eval/paper1/constraint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # rows[(task, model, mode)][base] = (exec, strict)
    rows = defaultdict(dict)
    cap_of = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "paper1-*.json"))):
        name = os.path.basename(path).replace(".json", "")
        m = re.match(r"paper1-(2B|4B|9B)-s(\d+)", name)
        if not m:
            continue
        cap_of[name] = m.group(1)
        pred = os.path.splitext(path)[0] + "_predictions.jsonl"
        if not os.path.isfile(pred):
            continue
        for line in open(pred, encoding="utf-8"):
            r = json.loads(line)
            rows[(r["task"], name, r["mode"])][r["base_id"]] = (int(bool(r["exec"])), int(bool(r["strict"])), int(bool(r["coverage"])), int(bool(r["fallback"])))
    summary = {}
    for task in TASKS:
        models = sorted({k[1] for k in rows if k[0] == task})
        if not models:
            continue
        per_model = {}
        for model in models:
            per_model[model] = {}
            for mode in MODES:
                vals = rows.get((task, model, mode), {})
                if not vals:
                    continue
                per_model[model][mode] = {
                    "n": len(vals),
                    "exec": round(statistics.mean(v[0] for v in vals.values()), 6),
                    "strict": round(statistics.mean(v[1] for v in vals.values()), 6),
                    "coverage_rate": round(statistics.mean(v[2] for v in vals.values()), 6),
                    "fallback_rate": round(statistics.mean(v[3] for v in vals.values()), 6),
                }
        contrasts = {}
        for mode in ("authorized", "postfilter", "random_size"):
            ids = sorted(set.intersection(
                *[set(rows[(task, m, "open")]) for m in models],
                *[set(rows[(task, m, mode)]) for m in models],
            ))
            ds = []
            for sid in ids:
                o = statistics.mean(rows[(task, m, "open")][sid][0] for m in models)
                x = statistics.mean(rows[(task, m, mode)][sid][0] for m in models)
                ds.append(x - o)
            contrasts[f"{mode}_minus_open"] = {
                "delta": round(statistics.mean(ds), 6) if ds else 0.0,
                "ci": [round(v, 6) for v in boot_ci(ds)] if ds else [0.0, 0.0],
                "n_base": len(ds),
            }
        # authorized vs random_size (composition vs size)
        ids = sorted(set.intersection(
            *[set(rows[(task, m, "authorized")]) for m in models],
            *[set(rows[(task, m, "random_size")]) for m in models],
        ))
        ds = []
        for sid in ids:
            a = statistics.mean(rows[(task, m, "authorized")][sid][0] for m in models)
            r = statistics.mean(rows[(task, m, "random_size")][sid][0] for m in models)
            ds.append(a - r)
        contrasts["authorized_minus_random_size"] = {
            "delta": round(statistics.mean(ds), 6) if ds else 0.0,
            "ci": [round(v, 6) for v in boot_ci(ds)] if ds else [0.0, 0.0],
            "n_base": len(ds),
        }
        by_cap = {}
        for cap in ("2B", "4B", "9B"):
            ms = [m for m in models if cap_of.get(m) == cap]
            if not ms:
                continue
            ids = sorted(set.intersection(*[set(rows[(task, m, "open")]) for m in ms],
                                           *[set(rows[(task, m, "authorized")]) for m in ms]))
            ds = []
            for sid in ids:
                o = statistics.mean(rows[(task, m, "open")][sid][0] for m in ms)
                a = statistics.mean(rows[(task, m, "authorized")][sid][0] for m in ms)
                ds.append(a - o)
            by_cap[cap] = {"authorized_minus_open": round(statistics.mean(ds), 6) if ds else 0.0,
                           "ci": [round(v, 6) for v in boot_ci(ds)] if ds else [0.0, 0.0],
                           "n_models": len(ms), "n_base": len(ds)}
        summary[task] = {"models": models, "per_model": per_model, "contrasts": contrasts, "by_capacity": by_cap}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    for task, s in summary.items():
        print("==", task, "n", s["contrasts"]["authorized_minus_open"]["n_base"])
        for k, v in s["contrasts"].items():
            print(" ", k, v["delta"], v["ci"])
        print("  cap", {k: round(v["authorized_minus_open"], 4) for k, v in s["by_capacity"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
