#!/usr/bin/env python3
"""Merge teacher raw batches, validate, dedupe, and split into train/val/test.

Split is stratified by ``generator`` with a fixed seed. The test split is
written once and should be treated as frozen (record its sha256).
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biod.multitask.catalog import load_catalog_snapshot
from biod.multitask.data import parse_sample
from biod.multitask.tasks import TASKS

SPLITS = ("train", "val", "test")


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", (text or "").lower())


def row_key(task: str, rec: dict) -> str:
    if task == "context_retention":
        return "|".join([norm(rec.get("state_text", ""))] + [norm(g.get("text", "")) for g in rec.get("groups", [])])
    if task == "relay_select":
        return "|".join([rec.get("from", ""), rec.get("to", ""), norm(rec.get("why", "")), norm(rec.get("body", ""))])
    if task == "premise_judge":
        return "|".join([rec.get("unit", ""), norm(rec.get("left", "")), norm(rec.get("right", ""))])
    if task == "overclaim_flag":
        return "|".join([norm(rec.get("claim", "")), norm(rec.get("evidence", ""))])
    return norm(rec.get("state_text", "")) + "|" + str((rec.get("label") or {}).get("role", ""))


def load_raw(raw_dir: str, task: str, catalog) -> list[dict]:
    files = sorted(glob.glob(os.path.join(raw_dir, task, "*.jsonl")))
    rows: list[dict] = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                parse_sample(rec, task, catalog)  # strict validation
                rows.append(rec)
    print(f"[raw] {task}: files={len(files)} rows={len(rows)}")
    return rows


def split_rows(rows: list[dict], task: str, seed: int) -> dict[str, list[dict]]:
    by_gen = defaultdict(list)
    for rec in rows:
        by_gen[str(rec.get("generator", "unknown"))].append(rec)
    out = {s: [] for s in SPLITS}
    rng = random.Random(seed)
    for gen, group in sorted(by_gen.items()):
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(0.10 * n)) if n >= 5 else (1 if n >= 3 else 0)
        n_val = max(1, round(0.10 * n)) if n >= 5 else (1 if n >= 3 else 0)
        if n_test + n_val >= n:
            n_test = 1 if n >= 2 else 0
            n_val = 0
        out["test"].extend(group[:n_test])
        out["val"].extend(group[n_test:n_test + n_val])
        out["train"].extend(group[n_test + n_val:])
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    catalog = load_catalog_snapshot()
    manifest = {"schemaVersion": 1, "seed": args.seed, "catalogHash": catalog.sha256, "tasks": {}}
    for task in (args.tasks or TASKS):
        rows = load_raw(args.raw_dir, task, catalog)
        if not rows:
            print(f"[skip] {task}: no raw rows")
            continue
        seen = set()
        unique = []
        for rec in rows:
            key = row_key(task, rec)
            if key in seen:
                continue
            seen.add(key)
            unique.append(rec)
        parts = split_rows(unique, task, args.seed)
        meta = {"raw": len(rows), "unique": len(unique)}
        for split in SPLITS:
            part = parts[split]
            path = os.path.join(args.out_dir, f"{task}_{split}.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for rec in part:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            meta[split] = {"rows": len(part), "sha256": sha256(path)}
        manifest["tasks"][task] = meta
        print(f"[split] {task}: unique={len(unique)} " + " ".join(f"{s}={len(parts[s])}" for s in SPLITS))
        print("        generators:", dict(Counter(str(r.get('generator')) for r in unique)))
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("[done]", os.path.join(args.out_dir, "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
