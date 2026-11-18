#!/usr/bin/env python3
"""Cross-fitted temperature scaling for the 7-head decision layer."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import hashlib
import statistics
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from biod.multitask.catalog import load_catalog_snapshot
from biod.multitask.data import load_split


def ece(probs, labels, bins=10):
    bucket = [[0, 0.0, 0.0] for _ in range(bins)]
    for p, y in zip(probs, labels):
        b = min(bins - 1, int(p * bins))
        bucket[b][0] += 1
        bucket[b][1] += p
        bucket[b][2] += y
    return sum(abs(c[2] / c[0] - c[1] / c[0]) * c[0] / len(probs) for c in bucket if c[0]) if probs else 0.0


def nll(probs, labels):
    eps = 1e-9
    return -sum(math.log(max(eps, p if y else 1 - p)) for p, y in zip(probs, labels)) / max(1, len(probs))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def temp_binary(logits, labels, T):
    return [sigmoid(x / T) for x in logits]


def temp_multi(logits, T):
    z = [x / T for x in logits]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def fit_T_binary(logits, labels):
    def obj(T):
        return nll(temp_binary(logits, labels, T), labels)
    r = minimize_scalar(obj, bounds=(0.05, 5.0), method="bounded")
    return float(r.x)


def fit_T_multi(logits_rows, labels):
    # logits_rows: list of list; labels: list of index
    def obj(T):
        ps = [temp_multi(row, T) for row in logits_rows]
        return -sum(math.log(max(1e-9, p[y])) for p, y in zip(ps, labels)) / max(1, len(labels))
    r = minimize_scalar(obj, bounds=(0.05, 5.0), method="bounded")
    return float(r.x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/eval/paper1/models")
    ap.add_argument("--data-dir", default="Datasets/paper1/legacy-test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    catalog = load_catalog_snapshot()
    gold = {t: {s.id: s for s in load_split(args.data_dir, t, "test", catalog)}
            for t in ("tool_select", "kb_select", "role_select", "relay_select", "premise_judge", "overclaim_flag")}
    # labels/prob vectors
    rows = defaultdict(dict)  # (task, model) -> base_id -> (probs, labels, kind)
    for path in sorted(glob.glob(os.path.join(args.dir, "paper1-*/legacy_7task_predictions.jsonl"))):
        name = path.split("/")[-2]
        if "lr2e5" in name:
            continue
        for line in open(path):
            r = json.loads(line)
            task = r["task"]
            if task not in gold:
                continue
            sid = r["id"]
            s = gold[task].get(sid)
            if s is None:
                continue
            if task == "tool_select":
                probs = r.get("probs") or {}
                ids = list(probs)
                if not ids:
                    continue
                y = [1.0 if i == (s.label or {}).get("id") else 0.0 for i in ids]
                rows[(task, name)][sid] = ([probs[i] for i in ids], y, "binary")
            elif task == "kb_select":
                scores = r.get("scores") or {}
                ids = list(scores)
                if not ids:
                    continue
                grades = (s.label or {}).get("grades") or {}
                y = [1.0 if grades.get(i) == 2 else 0.0 for i in ids]
                rows[(task, name)][sid] = ([scores[i] for i in ids], y, "binary")
            elif task == "role_select":
                probs = r.get("probs") or {}
                ids = list(probs)
                if not ids:
                    continue
                g = (s.label or {}).get("role")
                y = [1.0 if i == g else 0.0 for i in ids]
                rows[(task, name)][sid] = ([probs[i] for i in ids], y, "multi")
            elif task == "relay_select":
                probs = r.get("probs") or {}
                ids = list(probs)
                if not ids:
                    continue
                g = (s.label or {}).get("verdict")
                y = [1.0 if i == g else 0.0 for i in ids]
                rows[(task, name)][sid] = ([probs[i] for i in ids], y, "multi")
            elif task == "premise_judge":
                probs = r.get("probs") or {}
                ids = list(probs)
                if not ids:
                    continue
                g = (s.label or {}).get("relation")
                y = [1.0 if i == g else 0.0 for i in ids]
                rows[(task, name)][sid] = ([probs[i] for i in ids], y, "multi")
            elif task == "overclaim_flag":
                probs = r.get("probs") or {}
                ids = list(probs)
                if not ids:
                    continue
                flags = set((s.label or {}).get("flags") or [])
                y = [1.0 if i in flags else 0.0 for i in ids]
                rows[(task, name)][sid] = ([probs[i] for i in ids], y, "binary")

    out = {}
    for (task, model), data in sorted(rows.items()):
        sids = sorted(data)
        # 5-fold by base id
        fold = {sid: int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % 5 for sid in sids}
        raw_p = []
        raw_y = []
        temp_p = []
        temp_y = []
        Ts = []
        for k in range(5):
            train = [sid for sid in sids if fold[sid] != k]
            test = [sid for sid in sids if fold[sid] == k]
            if not train or not test:
                continue
            if data[sids[0]][2] == "binary":
                # rows may have different lengths (candidate sets); fit per-row temperatures and average
                ts = []
                for sid in train:
                    p, y, _ = data[sid]
                    logits = [math.log(max(1e-6, min(1 - 1e-6, x))) - math.log(max(1e-6, 1 - min(1 - 1e-6, x))) for x in p]
                    ts.append(fit_T_binary(logits, y))
                T = statistics.median(ts) if ts else 1.0
                Ts.append(T)
                for sid in test:
                    p, y, _ = data[sid]
                    logits = [math.log(max(1e-6, min(1 - 1e-6, x))) - math.log(max(1e-6, 1 - min(1 - 1e-6, x))) for x in p]
                    raw_p.extend(p); raw_y.extend(y); temp_p.extend(temp_binary(logits, y, T)); temp_y.extend(y)
            else:
                # multiclass: pad? fit one T on all train rows
                train_rows = [[math.log(max(1e-9, x)) for x in data[sid][0]] for sid in train]
                train_y = [max(range(len(y)), key=lambda i: y[i]) for _, y, _ in (data[sid] for sid in train)]
                T = fit_T_multi(train_rows, train_y)
                Ts.append(T)
                for sid in test:
                    p, y, _ = data[sid]
                    raw_p.extend(p); raw_y.extend(y)
                    temp_p.extend(temp_multi([math.log(max(1e-9, x)) for x in p], T)); temp_y.extend(y)
        out[f"{task}::{model}"] = {
            "task": task, "model": model, "n_base": len(sids), "n_obs": len(raw_p),
            "T_median": round(statistics.median(Ts), 4) if Ts else None,
            "ece_raw": round(ece(raw_p, raw_y), 6), "ece_temp": round(ece(temp_p, temp_y), 6),
        }
    # aggregate per task
    agg = {}
    for task in sorted({v["task"] for v in out.values()}):
        vals = [v for v in out.values() if v["task"] == task]
        agg[task] = {
            "n_models": len(vals),
            "ece_raw": round(statistics.mean(v["ece_raw"] for v in vals), 6),
            "ece_temp": round(statistics.mean(v["ece_temp"] for v in vals), 6),
            "T_median": round(statistics.median(v["T_median"] for v in vals if v["T_median"]), 4),
        }
    doc = {"per_model": out, "aggregate": agg}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(json.dumps(agg, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
