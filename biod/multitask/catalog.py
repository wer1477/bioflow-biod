"""Role-conditioned catalog snapshot for the five decisions.

The snapshot is generated from the live bioagent-dev catalog (tool library +
knowledge base + agents.json bindings) by
``research/biod-5dp/scripts/build_catalog_snapshot.py`` and copied to
``Datasets/v2/catalog_snapshot.json``. Training and inference must use
the same snapshot; the hash is recorded in the run manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable

from .tasks import ROLES


@dataclass(frozen=True)
class CatalogOption:
    """One candidate rendered into a prompt / scored by a head."""

    id: str
    name: str
    description: str
    kind: str = "skill"
    domain: str = ""

    def render_text(self) -> str:
        # The pointer head only needs a per-candidate representation; the id is
        # carried by the slot mapping, so the prompt stays compact enough for a
        # 9B forward on a 48GB card. The generative baseline uses a separate
        # prompt that includes ids.
        desc = self.description.strip()
        if len(desc) > 100:
            desc = desc[:100].rstrip() + "…"
        return f"{self.name} — {desc}"


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class CatalogSnapshot:
    """Immutable view of tools, knowledge entries, and role visibility."""

    def __init__(self, raw: dict, path: str = ""):
        self.raw = raw
        self.path = path
        self.version = str(raw.get("catalogVersion") or raw.get("schemaVersion") or "")
        self.tools: dict[str, CatalogOption] = {
            item["id"]: CatalogOption(
                id=item["id"],
                name=item["name"],
                description=item.get("description", ""),
                kind=item.get("kind", "skill"),
                domain=item.get("domain", ""),
            )
            for item in raw.get("tools", [])
        }
        self.knowledge: dict[str, CatalogOption] = {
            item["id"]: CatalogOption(
                id=item["id"],
                name=item["name"],
                description=item.get("description", ""),
                kind="kb",
                domain=item.get("domain", ""),
            )
            for item in raw.get("knowledge", [])
        }
        self.roles: dict[str, dict] = raw.get("roles", {})
        for role in ROLES:
            if role not in self.roles:
                raise ValueError(f"catalog snapshot is missing role {role!r}")
        self.sha256 = self._hash()

    def _hash(self) -> str:
        blob = json.dumps(self.raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # -- role views ---------------------------------------------------------
    def role_tool_ids(self, role: str) -> list[str]:
        return list(self.roles[role]["visibleToolIds"])

    def role_knowledge_ids(self, role: str) -> list[str]:
        return list(self.roles[role]["visibleKnowledgeIds"])

    def tool_options(self, role: str | None = None, ids: Iterable[str] | None = None) -> list[CatalogOption]:
        """Canonical-order options for a role (or an explicit id list)."""
        if ids is not None:
            return [self.tools[i] for i in ids]
        role = role or "pi"
        return [self.tools[i] for i in self.role_tool_ids(role)]

    def knowledge_options(self, role: str | None = None, ids: Iterable[str] | None = None) -> list[CatalogOption]:
        if ids is not None:
            return [self.knowledge[i] for i in ids]
        role = role or "pi"
        return [self.knowledge[i] for i in self.role_knowledge_ids(role)]

    def validate_candidates(self, task: str, role: str, candidate_ids: Iterable[str]) -> list[str]:
        ids = list(candidate_ids)
        expected = set(self.role_tool_ids(role) if task == "tool_select" else self.role_knowledge_ids(role))
        unknown = [i for i in ids if i not in (self.tools if task == "tool_select" else self.knowledge)]
        if unknown:
            raise ValueError(f"{task}/{role}: unknown catalog ids {unknown[:5]}")
        missing = expected - set(ids)
        extra = set(ids) - expected
        if missing or extra:
            raise ValueError(
                f"{task}/{role}: candidates must equal the role-visible set "
                f"(missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
            )
        return ids


def load_catalog_snapshot(path: str | None = None) -> CatalogSnapshot:
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, "Datasets", "v2", "catalog_snapshot.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"catalog snapshot not found: {path}; run "
            "research/biod-5dp/scripts/build_catalog_snapshot.py and copy it to "
            "Datasets/v2/catalog_snapshot.json"
        )
    return CatalogSnapshot(_read_json(path), path=path)
