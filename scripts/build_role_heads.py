#!/usr/bin/env python
"""Merge, expand, validate and dedupe the role-head teacher datasets.

Reads ``Datasets/v3/role_heads/raw/<task>/*.jsonl`` (teacher output),
expands ``@preset:<role>`` candidate lists, validates every record, drops
duplicates/invalid rows with an explicit reason, and writes a merged file per
task plus a stats report.

Usage:
    .venv/bin/python scripts/build_role_heads.py [--raw-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from biod.multitask.catalog import load_catalog_snapshot  # noqa: E402
from biod.multitask.data import parse_sample  # noqa: E402

RELATIONS = ("supports", "partially_supports", "contradicts", "unrelated", "insufficient")
UNITS = ("hypothesis", "claim", "issue")
FLAGS = ("scope", "causality", "certainty", "generalization", "temporal")
OWNER_REASONS = (
    "owns_new_experiment",
    "owns_new_search",
    "owns_manuscript_edit",
    "owns_review_judgement",
    "needs_arbitration",
    "out_of_scope",
)
LIMITS = {
    "premise_judge": {"left": 700, "right": 1400, "context": 400},
    "overclaim_flag": {"claim": 400, "evidence": 1200, "context": 300},
    "tool_select": {"state_text": 780},
    "role_select": {"state_text": 1200},
}
TASKS = ("premise_judge", "overclaim_flag", "tool_select", "role_select")


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def dedupe_key(task: str, rec: dict) -> str:
    if task == "premise_judge":
        return f"{rec.get('unit')}::{norm(rec.get('left', ''))}::{norm(rec.get('right', ''))}"
    if task == "overclaim_flag":
        return f"{norm(rec.get('claim', ''))}::{norm(rec.get('evidence', ''))}"
    return norm(rec.get("state_text", ""))


def expand_presets(task: str, rec: dict, catalog, errors: list[str]) -> None:
    ids = rec.get("candidateIds")
    if not isinstance(ids, list) or len(ids) != 1 or not str(ids[0]).startswith("@preset:"):
        return
    role = str(ids[0]).split(":", 1)[1]
    try:
        rec["candidateIds"] = catalog.role_tool_ids(role)
        rec["role"] = role
    except Exception as exc:  # noqa: BLE001
        errors.append(f"preset {role!r} failed: {exc}")


def check_common(task: str, rec: dict, seen_ids: set[str], errors: list[str]) -> bool:
    rid = rec.get("id")
    if not isinstance(rid, str) or not rid:
        errors.append("missing id")
        return False
    if rid in seen_ids:
        errors.append(f"{rid}: duplicate id")
        return False
    seen_ids.add(rid)
    if rec.get("task") != task:
        errors.append(f"{rid}: task mismatch {rec.get('task')!r}")
        return False
    if rec.get("source") != "teacher_synthetic":
        errors.append(f"{rid}: source must be teacher_synthetic")
        return False
    if rec.get("difficulty") not in ("easy", "medium", "hard"):
        errors.append(f"{rid}: bad difficulty")
        return False
    if rec.get("lang") not in ("zh", "en"):
        errors.append(f"{rid}: bad lang")
        return False
    for field, cap in LIMITS[task].items():
        value = rec.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{rid}: empty {field}")
            return False
        if len(value) > cap:
            errors.append(f"{rid}: {field} too long ({len(value)}>{cap})")
            return False
    return True


def check_premise(rec: dict, errors: list[str]) -> bool:
    rid = rec.get("id")
    if rec.get("unit") not in UNITS:
        errors.append(f"{rid}: bad unit {rec.get('unit')!r}")
        return False
    label = rec.get("label") or {}
    if label.get("relation") not in RELATIONS:
        errors.append(f"{rid}: bad relation {label.get('relation')!r}")
        return False
    conf = label.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.5 <= float(conf) <= 1.0):
        errors.append(f"{rid}: bad confidence {conf!r}")
        return False
    return True


def check_overclaim(rec: dict, errors: list[str]) -> bool:
    rid = rec.get("id")
    if rec.get("generator") not in ("overclaim", "clean"):
        errors.append(f"{rid}: bad generator {rec.get('generator')!r}")
        return False
    label = rec.get("label") or {}
    flags = label.get("flags")
    if not isinstance(flags, list) or any(f not in FLAGS for f in flags):
        errors.append(f"{rid}: bad flags {flags!r}")
        return False
    if len(set(flags)) != len(flags):
        errors.append(f"{rid}: duplicate flags")
        return False
    if rec.get("generator") == "clean" and flags:
        errors.append(f"{rid}: clean sample has flags")
        return False
    if rec.get("generator") == "overclaim" and not flags:
        errors.append(f"{rid}: overclaim sample has no flags")
        return False
    conf = label.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.5 <= float(conf) <= 1.0):
        errors.append(f"{rid}: bad confidence {conf!r}")
        return False
    return True


def check_role(rec: dict, catalog, errors: list[str]) -> bool:
    rid = rec.get("id")
    if rec.get("generator") != "issue_owner":
        errors.append(f"{rid}: bad generator {rec.get('generator')!r}")
        return False
    label = rec.get("label") or {}
    if label.get("reason_code") not in OWNER_REASONS:
        errors.append(f"{rid}: bad reason_code {label.get('reason_code')!r}")
        return False
    try:
        parse_sample(rec, "role_select", catalog)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{rid}: {exc}")
        return False
    return True


def check_tool(rec: dict, catalog, errors: list[str]) -> bool:
    rid = rec.get("id")
    if rec.get("generator") != "plan_step":
        errors.append(f"{rid}: bad generator {rec.get('generator')!r}")
        return False
    try:
        parse_sample(rec, "tool_select", catalog)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{rid}: {exc}")
        return False
    return True


def summarize(recs: list[dict], task: str) -> dict:
    def dist(fn):
        counts = collections.Counter(fn(r) for r in recs)
        return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in counts.most_common()}

    lengths = []
    for r in recs:
        for field in LIMITS[task]:
            lengths.append(len(str(r.get(field, ""))))
    return {
        "count": len(recs),
        "labels": dist(
            lambda r: r["label"].get("relation")
            if task == "premise_judge"
            else (tuple(r["label"].get("flags", [])) if task == "overclaim_flag"
                  else (r["label"].get("role") or r["label"].get("id")))
        ),
        "generators": dist(lambda r: r.get("generator")),
        "lang": dist(lambda r: r.get("lang")),
        "difficulty": dist(lambda r: r.get("difficulty")),
        "len_p50": int(statistics.median(lengths)) if lengths else 0,
        "len_p95": int(sorted(lengths)[int(len(lengths) * 0.95)]) if lengths else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(REPO, "Datasets/v3/role_heads/raw"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "Datasets/v3/role_heads/merged"))
    args = ap.parse_args()

    catalog = load_catalog_snapshot()
    os.makedirs(args.out_dir, exist_ok=True)
    report: dict = {}
    total_errors = 0

    for task in TASKS:
        files = sorted(glob.glob(os.path.join(args.raw_dir, task, "*.jsonl")))
        recs: list[dict] = []
        seen: set[str] = set()
        keys: set[str] = set()
        errors: list[str] = []
        dropped = collections.Counter()

        for path in files:
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"{os.path.basename(path)}:{lineno}: bad json ({exc})")
                        dropped["bad_json"] += 1
                        continue
                    local_errors: list[str] = []
                    expand_presets(task, rec, catalog, local_errors)
                    ok = not local_errors and check_common(task, rec, seen, local_errors)
                    if ok:
                        if task == "premise_judge":
                            ok = check_premise(rec, local_errors)
                        elif task == "overclaim_flag":
                            ok = check_overclaim(rec, local_errors)
                        elif task == "role_select":
                            ok = check_role(rec, catalog, local_errors)
                        else:
                            ok = check_tool(rec, catalog, local_errors)
                    if not ok:
                        errors.extend(local_errors)
                        dropped["invalid"] += 1
                        continue
                    key = dedupe_key(task, rec)
                    if key in keys:
                        dropped["duplicate"] += 1
                        continue
                    keys.add(key)
                    recs.append(rec)

        task_dir = os.path.join(args.out_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        out_path = os.path.join(task_dir, "merged.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats = summarize(recs, task)
        stats["files"] = len(files)
        stats["dropped"] = dict(dropped)
        stats["errors"] = len(errors)
        report[task] = stats
        total_errors += len(errors)
        print(f"[{task}] kept={stats['count']} files={len(files)} dropped={dict(dropped)} errors={len(errors)}")
        for err in errors[:5]:
            print(f"    ! {err}")

    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: {"count": v["count"], "labels": v["labels"]} for k, v in report.items()},
                     ensure_ascii=False, indent=2))
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
