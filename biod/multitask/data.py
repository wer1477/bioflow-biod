"""JSONL loaders for the five-decision dataset.

The teacher emits compact samples (candidate ids, not full descriptions); this
module joins them with the catalog snapshot and validates every invariant the
training pipeline relies on. A malformed record must fail loudly: silently
shifting a label slot is the worst possible bug in a pointer head.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .catalog import CatalogOption, CatalogSnapshot
from .tasks import (
    CONTEXT_LABELS,
    OVERCLAIM_FLAGS,
    PREMISE_LABELS,
    PREMISE_UNITS,
    RELAY_REASONS,
    RELAY_VERDICTS,
    ROLES,
    TASK_SPECS,
)

NO_TOOL = "NO_TOOL"


@dataclass
class Sample:
    task: str
    id: str
    role: str | None
    state_text: str
    options: list[CatalogOption] = field(default_factory=list)
    # tool: gold id; kb: grade map; context: label list; role: role id; relay: verdict
    label: object = None
    from_role: str | None = None
    to_role: str | None = None
    why: str = ""
    body: str = ""
    groups: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    left: str = ""
    right: str = ""
    context: str = ""
    unit: str = ""

    @property
    def num_slots(self) -> int:
        return len(self.options) + (1 if self.task == "tool_select" else 0)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _check_role(role: str, rid: str) -> str:
    _require(role in ROLES, f"{rid}: bad role {role!r}")
    return role


def parse_sample(rec: dict, task: str, catalog: CatalogSnapshot) -> Sample:
    rid = str(rec.get("id") or "<unknown>")
    _require(rec.get("task") == task, f"{rid}: task {rec.get('task')!r} != {task!r}")

    if task in {"tool_select", "kb_select"}:
        role = _check_role(str(rec.get("role")), rid)
        candidate_ids = list(rec.get("candidateIds") or [])
        catalog.validate_candidates(task, role, candidate_ids)
        options = (
            catalog.tool_options(ids=candidate_ids)
            if task == "tool_select"
            else catalog.knowledge_options(ids=candidate_ids)
        )
        label = rec.get("label")
        _require(isinstance(label, dict), f"{rid}: missing label object")
        if task == "tool_select":
            gold = str(label.get("id", ""))
            _require(gold == NO_TOOL or gold in candidate_ids, f"{rid}: label id {gold!r} not a candidate")
        else:
            grades = label.get("grades") or {}
            _require(set(grades) == set(candidate_ids), f"{rid}: grades must cover every candidate")
            _require(all(v in (0, 1, 2) for v in grades.values()), f"{rid}: grade outside 0/1/2")
            selected = label.get("selected") or []
            _require(len(selected) <= 3, f"{rid}: selected > 3")
            _require(set(selected) <= set(candidate_ids), f"{rid}: selected has unknown id")
            _require(all(grades.get(x) == 2 for x in selected), f"{rid}: selected must be grade 2")
        return Sample(
            task=task, id=rid, role=role, state_text=str(rec.get("state_text", "")),
            options=options, label=label,
            meta={k: rec.get(k) for k in ("source", "generator", "difficulty", "lang") if k in rec},
        )

    if task == "context_retention":
        groups = list(rec.get("groups") or [])
        labels = list(rec.get("label") or [])
        _require(len(groups) >= 1, f"{rid}: no groups")
        _require(len(labels) == len(groups), f"{rid}: label count != group count")
        gseq = [int(g["seq"]) for g in groups]
        lseq = [int(x["seq"]) for x in labels]
        _require(gseq == lseq, f"{rid}: label seqs {lseq} != group seqs {gseq}")
        for g in groups:
            _require(g.get("kind") != "user", f"{rid}: user groups are not scored")
            _require(str(g.get("text", "")).strip(), f"{rid}: empty group text")
        for x in labels:
            _require(x.get("label") in CONTEXT_LABELS, f"{rid}: bad context label {x.get('label')!r}")
        return Sample(
            task=task, id=rid, role=None, state_text=str(rec.get("state_text", "")),
            groups=groups, label=labels,
            meta={k: rec.get(k) for k in ("source", "generator", "difficulty", "lang") if k in rec},
        )

    if task == "role_select":
        label = rec.get("label") or {}
        role = _check_role(str(label.get("role")), rid)
        conf = label.get("confidence")
        _require(isinstance(conf, (int, float)) and 0 <= conf <= 1, f"{rid}: bad confidence {conf!r}")
        return Sample(
            task=task, id=rid, role=None, state_text=str(rec.get("state_text", "")),
            label={"role": role, "confidence": float(conf), "reason_code": label.get("reason_code")},
            meta={k: rec.get(k) for k in ("source", "generator", "difficulty", "lang") if k in rec},
        )

    if task == "relay_select":
        from_role = _check_role(str(rec.get("from")), rid)
        to_role = _check_role(str(rec.get("to")), rid)
        _require(from_role != to_role, f"{rid}: from == to")
        label = rec.get("label") or {}
        verdict = str(label.get("verdict", ""))
        _require(verdict in RELAY_VERDICTS, f"{rid}: bad verdict {verdict!r}")
        reason = label.get("reason_code")
        _require(reason in RELAY_REASONS, f"{rid}: bad reason {reason!r}")
        return Sample(
            task=task, id=rid, role=None, state_text="",
            from_role=from_role, to_role=to_role,
            why=str(rec.get("why", "")), body=str(rec.get("body", "")),
            label={"verdict": verdict, "confidence": float(label.get("confidence", 0.0)), "reason_code": reason},
            meta={k: rec.get(k) for k in ("source", "generator", "difficulty", "lang") if k in rec},
        )

    if task == "premise_judge":
        unit = str(rec.get("unit") or "")
        _require(unit in PREMISE_UNITS, f"{rid}: bad unit {unit!r}")
        left = str(rec.get("left") or "").strip()
        right = str(rec.get("right") or "").strip()
        _require(bool(left) and bool(right), f"{rid}: empty left/right")
        label = rec.get("label") or {}
        relation = str(label.get("relation", ""))
        _require(relation in PREMISE_LABELS, f"{rid}: bad relation {relation!r}")
        conf = label.get("confidence")
        _require(isinstance(conf, (int, float)) and 0 <= conf <= 1, f"{rid}: bad confidence {conf!r}")
        return Sample(
            task=task, id=rid, role=None, state_text="",
            label={"relation": relation, "confidence": float(conf)},
            left=left, right=right, context=str(rec.get("context") or ""), unit=unit,
            meta={k: rec.get(k) for k in ("source", "generator", "difficulty", "lang") if k in rec},
        )

    if task == "overclaim_flag":
        claim = str(rec.get("claim") or "").strip()
        evidence = str(rec.get("evidence") or "").strip()
        _require(bool(claim) and bool(evidence), f"{rid}: empty claim/evidence")
        label = rec.get("label") or {}
        flags = list(label.get("flags") or [])
        _require(all(f in OVERCLAIM_FLAGS for f in flags), f"{rid}: bad flags {flags!r}")
        _require(len(set(flags)) == len(flags), f"{rid}: duplicate flags")
        conf = label.get("confidence")
        _require(isinstance(conf, (int, float)) and 0 <= conf <= 1, f"{rid}: bad confidence {conf!r}")
        return Sample(
            task=task, id=rid, role=None, state_text="",
            label={"flags": flags, "confidence": float(conf)},
            left=claim, right=evidence, context=str(rec.get("context") or ""),
            meta={k: rec.get(k) for k in ("source", "generator", "difficulty", "lang") if k in rec},
        )

    raise ValueError(f"{rid}: unknown task {task!r}")


def load_task(
    path: str,
    task: str,
    catalog: CatalogSnapshot,
    limit: int = 0,
) -> list[Sample]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out: list[Sample] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            out.append(parse_sample(rec, task, catalog))
            if limit and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"{path}: empty dataset")
    return out


def split_path(data_dir: str, task: str, split: str) -> str:
    return os.path.join(data_dir, f"{task}_{split}.jsonl")


def load_split(
    data_dir: str,
    task: str,
    split: str,
    catalog: CatalogSnapshot,
    limit: int = 0,
) -> list[Sample]:
    return load_task(split_path(data_dir, task, split), task, catalog, limit=limit)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(data_dir: str, tasks: Iterable[str] = TASK_SPECS) -> dict:
    manifest: dict = {"data_dir": os.path.abspath(data_dir), "catalogHash": "", "splits": {}}
    for task in tasks:
        manifest["splits"][task] = {}
        for split in ("train", "val", "test"):
            path = split_path(data_dir, task, split)
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    n = sum(1 for _ in fh)
                manifest["splits"][task][split] = {"rows": n, "sha256": sha256_file(path)}
    return manifest
