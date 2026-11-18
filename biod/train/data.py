"""JSONL loaders + catalog validation for SFT v0 data.

Schema: ``docs/SFT-v0-执行契约.md`` §2. The loader is strict on purpose — a bad
candidate list silently shifts every label slot, so it must fail loudly.

* tool record: ``candidates`` must equal the frozen tool catalog ids **in
  canonical order**; ``gold.primary`` ∈ candidates ∪ {``NO_TOOL``}.
* kb record: ``candidates`` must equal the frozen kb catalog ids in canonical
  order; ``grades`` keys ⊆ candidates, values ∈ {0,1,2}; entries absent from
  ``grades`` are implicitly 0 (closed world). Soft label = ``grade / 2``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable

from biod.catalog import Option, catalog_hash, load_knowledge_base, load_tool_library

from .config import (
    KB_CATALOG_DIR,
    KB_CATALOG_VERSION,
    NO_TOOL_ID,
    TOOL_CATALOG_DIR,
    TOOL_CATALOG_VERSION,
    TOOL_NUM_OPTIONS,
    KB_NUM_OPTIONS,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Catalog:
    task: str  # "tool" | "kb"
    options: list[Option]
    hash: str
    version: str

    def __post_init__(self) -> None:
        expected_version = TOOL_CATALOG_VERSION if self.task == "tool" else KB_CATALOG_VERSION
        if self.version and self.version != expected_version:
            raise ValueError(f"{self.task} catalogVersion {self.version!r} != frozen {expected_version!r}")
        expected_n = TOOL_NUM_OPTIONS if self.task == "tool" else KB_NUM_OPTIONS
        if len(self.options) != expected_n:
            raise ValueError(f"{self.task} catalog has {len(self.options)} options, expected {expected_n}")

    @property
    def ids(self) -> list[str]:
        return [o.id for o in self.options]

    @property
    def id_to_index(self) -> dict[str, int]:
        return {o.id: i for i, o in enumerate(self.options)}


def _repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def load_catalog(task: str, root: str | None = None) -> Catalog:
    """Load the frozen catalog for ``task`` and verify its hash."""
    root = root or _REPO_ROOT
    if task == "tool":
        domains = load_tool_library(_repo_path(os.path.join(root, TOOL_CATALOG_DIR)) if not os.path.isabs(root) else os.path.join(root, TOOL_CATALOG_DIR))
        options = [o for d in domains for o in d.options]
        version = TOOL_CATALOG_VERSION
    elif task == "kb":
        domains = load_knowledge_base(_repo_path(os.path.join(root, KB_CATALOG_DIR)) if not os.path.isabs(root) else os.path.join(root, KB_CATALOG_DIR))
        options = [o for d in domains for o in d.options]
        version = KB_CATALOG_VERSION
    else:
        raise ValueError(f"unknown task {task!r}")
    digest = catalog_hash(options)
    expected = TOOL_CATALOG_VERSION if task == "tool" else KB_CATALOG_VERSION
    if digest != expected:
        raise ValueError(
            f"{task} catalog hash mismatch: {digest} != frozen {expected}; "
            "refusing to train on a moved catalog"
        )
    return Catalog(task=task, options=options, hash=digest, version=version)


def load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset file not found: {path}")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - data bug
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"dataset file is empty: {path}")
    return rows


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_tool_record(rec: dict, catalog: Catalog) -> dict:
    rid = rec.get("id") or rec.get("qid") or "<unknown>"
    _require(isinstance(rec.get("state_text"), str) and rec["state_text"].strip(), f"{rid}: missing state_text")
    cands = rec.get("candidates")
    _require(isinstance(cands, list), f"{rid}: missing candidates list")
    if cands != catalog.ids:
        raise ValueError(
            f"{rid}: candidates must equal the canonical {len(catalog.ids)}-tool catalog "
            f"(got {len(cands)} entries; first mismatch "
            f"{next((i for i, (a, b) in enumerate(zip(cands, catalog.ids)) if a != b), 'length')})"
        )
    gold = rec.get("gold") or {}
    primary = gold.get("primary")
    _require(isinstance(primary, str) and primary, f"{rid}: missing gold.primary")
    if primary == NO_TOOL_ID:
        variant = (rec.get("meta") or {}).get("variant")
        _require(
            variant in {"no_tool", "hard_neg", None},
            f"{rid}: NO_TOOL gold only allowed for variant no_tool|hard_neg, got {variant!r}",
        )
    else:
        _require(primary in catalog.id_to_index, f"{rid}: gold.primary {primary!r} not in catalog")
    acceptable = gold.get("acceptable") or []
    _require(isinstance(acceptable, list), f"{rid}: gold.acceptable must be a list")
    for aid in acceptable:
        _require(aid in catalog.id_to_index, f"{rid}: acceptable id {aid!r} not in catalog")
    out = dict(rec)
    out["id"] = rid
    out["task"] = "tool"
    out["gold_primary"] = primary
    out["gold_acceptable"] = list(acceptable)
    return out


def validate_kb_record(rec: dict, catalog: Catalog) -> dict:
    rid = rec.get("id") or rec.get("qid") or "<unknown>"
    _require(isinstance(rec.get("state_text"), str) and rec["state_text"].strip(), f"{rid}: missing state_text")
    cands = rec.get("candidates")
    _require(isinstance(cands, list), f"{rid}: missing candidates list")
    if cands != catalog.ids:
        raise ValueError(
            f"{rid}: candidates must equal the canonical {len(catalog.ids)}-entry kb catalog "
            f"(got {len(cands)} entries)"
        )
    grades = rec.get("grades")
    _require(isinstance(grades, dict), f"{rid}: missing grades dict")
    soft: dict[str, float] = {}
    for key, val in grades.items():
        _require(key in catalog.id_to_index, f"{rid}: grade id {key!r} not in catalog")
        _require(isinstance(val, (int, float)) and int(val) == val and 0 <= int(val) <= 2,
                 f"{rid}: grade for {key!r} must be int in 0..2, got {val!r}")
        soft[key] = int(val) / 2.0
    if (rec.get("meta") or {}).get("variant") == "near_miss":
        _require(any(v == 0.5 for v in soft.values()), f"{rid}: near_miss sample must contain an explicit grade-1 label")
    out = dict(rec)
    out["id"] = rid
    out["task"] = "kb"
    out["soft_labels"] = soft
    out["num_positive"] = sum(1 for v in soft.values() if v > 0)
    return out


def load_task_records(
    data_dir: str,
    task: str,
    split: str,
    catalog: Catalog,
    limit: int = 0,
) -> list[dict]:
    """Load + validate one split of one task. ``limit`` truncates (smoke only)."""
    fname = f"{task}_{split}.jsonl"
    path = os.path.join(_repo_path(data_dir), fname)
    rows = load_jsonl(path)
    validator = validate_tool_record if task == "tool" else validate_kb_record
    out = []
    for rec in rows:
        out.append(validator(rec, catalog))
        if limit and len(out) >= limit:
            break
    return out


def dataset_manifest_entry(data_dir: str, task: str, split: str, limit: int = 0) -> dict | None:
    path = os.path.join(_repo_path(data_dir), f"{task}_{split}.jsonl")
    if not os.path.isfile(path):
        return None
    return {"path": path, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}


def summarize_records(records: Iterable[dict], task: str) -> dict:
    records = list(records)
    summary: dict = {"count": len(records)}
    if task == "tool":
        summary["no_tool"] = sum(1 for r in records if r["gold_primary"] == NO_TOOL_ID)
        variants: dict[str, int] = {}
        for r in records:
            v = (r.get("meta") or {}).get("variant", "unknown")
            variants[v] = variants.get(v, 0) + 1
        summary["variants"] = variants
    else:
        summary["empty"] = sum(1 for r in records if r["num_positive"] == 0)
        summary["multi"] = sum(1 for r in records if r["num_positive"] > 1)
        summary["avg_positive"] = round(sum(r["num_positive"] for r in records) / max(1, len(records)), 3)
    return summary
