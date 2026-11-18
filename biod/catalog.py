"""Readers for bioflow-dev's two-level capability/knowledge catalogs.

The on-disk contract mirrors ``docs/toollibrary-mechanism.md`` and
``docs/knowledgebase-mechanism.md``:

tool library
    toollibrary/index.md                      ``- <domain>：<description>``
    toollibrary/<domain>/index.md             ``- <name>（skill：<id>）：<description>``

knowledge base
    knowledgebase/skill.md                    ``- <domain>：<description>``
    knowledgebase/<domain>/<entry>/skill.md   YAML frontmatter (name, description) + body

Both levels are read-only on disk; this module never writes.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

# Tolerance for full-width / half-width punctuation used in the markdown contract.
_DOMAIN_RE = re.compile(r"^\s*-\s*(?P<name>[^：:]+?)\s*[：:]\s*(?P<description>.+?)\s*$")
_ENTRY_RE = re.compile(
    r"^\s*-\s*(?P<name>[^（(]+?)\s*[（(]\s*(?P<kind>skill|mcp)\s*[：:]\s*"
    r"(?P<id>[^）)]+?)\s*[）)]\s*[：:]\s*(?P<description>.+?)\s*$"
)
_FRONTMATTER_RE = re.compile(r"^---\r?\n(?P<front>.*?)\r?\n---\r?\n?(?P<body>.*)$", re.S)


@dataclass
class Option:
    """One selectable candidate (a tool, an MCP tool, or a knowledge entry)."""

    index: int
    name: str
    description: str
    id: str
    kind: str = "skill"  # skill | mcp | kb
    layer: str = ""      # tool_l2 | kb_l2 | ...
    domain: str = ""

    def render_text(self) -> str:
        return f"{self.name} — {self.description}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "description": self.description,
            "id": self.id,
            "kind": self.kind,
            "layer": self.layer,
            "domain": self.domain,
        }


@dataclass
class Domain:
    name: str
    description: str
    options: list[Option] = field(default_factory=list)
    layer: str = ""
    kind: str = ""  # tool | kb

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "layer": self.layer,
            "kind": self.kind,
            "entryCount": len(self.options),
            "options": [o.to_dict() for o in self.options],
        }


# --------------------------------------------------------------------------- #
# tool library
# --------------------------------------------------------------------------- #
def read_domains(index_md: str) -> list[tuple[str, str]]:
    """Parse a level-1 index file into ``[(domain, description), ...]``."""
    out: list[tuple[str, str]] = []
    if not os.path.isfile(index_md):
        return out
    with open(index_md, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _DOMAIN_RE.match(line)
            if m:
                out.append((m.group("name").strip(), m.group("description").strip()))
    return out


def read_tool_catalog_domain(tool_root: str, domain: str) -> list[Option]:
    """Parse ``toollibrary/<domain>/index.md`` -> list of :class:`Option`."""
    entries: list[Option] = []
    index_md = os.path.join(tool_root, domain, "index.md")
    if not os.path.isfile(index_md):
        return entries
    with open(index_md, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _ENTRY_RE.match(line)
            if not m:
                continue
            entries.append(
                Option(
                    index=len(entries),
                    name=m.group("name").strip(),
                    description=m.group("description").strip(),
                    id=m.group("id").strip(),
                    kind=m.group("kind"),
                    layer="tool_l2",
                    domain=domain,
                )
            )
    return entries


def load_tool_library(tool_root: str) -> list[Domain]:
    """Load the full two-level tool library (domains with their options)."""
    domains: list[Domain] = []
    for name, desc in read_domains(os.path.join(tool_root, "index.md")):
        domains.append(
            Domain(
                name=name,
                description=desc,
                options=read_tool_catalog_domain(tool_root, name),
                layer="tool_l1",
                kind="tool",
            )
        )
    return domains


# --------------------------------------------------------------------------- #
# knowledge base
# --------------------------------------------------------------------------- #
def _parse_frontmatter(text: str) -> tuple[str, str, str]:
    """Return ``(name, description, body)`` from a knowledge ``skill.md``."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return "", "", text.strip()
    front, body = m.group("front"), m.group("body")
    name = ""
    description = ""
    for line in front.splitlines():
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")
    # description may be a folded YAML block; join continuation lines heuristically
    if description in ("|", ">", ""):
        description = ""
    return name, description, body.strip()


def read_kb_catalog_domain(kb_root: str, domain: str) -> list[Option]:
    """Parse ``knowledgebase/<domain>/<entry>/skill.md`` frontmatter."""
    entries: list[Option] = []
    domain_dir = os.path.join(kb_root, domain)
    if not os.path.isdir(domain_dir):
        return entries
    for entry in sorted(os.listdir(domain_dir)):
        if entry.startswith("."):
            continue
        md = os.path.join(domain_dir, entry, "skill.md")
        if not os.path.isfile(md):
            continue
        with open(md, encoding="utf-8", errors="replace") as fh:
            name, desc, _body = _parse_frontmatter(fh.read())
        entries.append(
            Option(
                index=len(entries),
                name=name or entry,
                description=desc,
                id=f"{domain}/{entry}",
                kind="kb",
                layer="kb_l2",
                domain=domain,
            )
        )
    return entries


def load_knowledge_base(kb_root: str) -> list[Domain]:
    domains: list[Domain] = []
    for name, desc in read_domains(os.path.join(kb_root, "skill.md")):
        domains.append(
            Domain(
                name=name,
                description=desc,
                options=read_kb_catalog_domain(kb_root, name),
                layer="kb_l1",
                kind="kb",
            )
        )
    # domains present on disk but missing from the index are still usable
    if os.path.isdir(kb_root):
        known = {d.name for d in domains}
        for entry in sorted(os.listdir(kb_root)):
            path = os.path.join(kb_root, entry)
            if entry.startswith(".") or not os.path.isdir(path) or entry in known:
                continue
            if entry == "resources":
                continue
            domains.append(
                Domain(
                    name=entry,
                    description="",
                    options=read_kb_catalog_domain(kb_root, entry),
                    layer="kb_l1",
                    kind="kb",
                )
            )
    return domains


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def domain_as_options(domains: Iterable[Domain], layer: str) -> list[Option]:
    """Turn level-1 domains into selectable options (name + description)."""
    return [
        Option(
            index=i,
            name=d.name,
            description=d.description,
            id=d.name,
            kind="domain",
            layer=layer,
            domain=d.name,
        )
        for i, d in enumerate(domains)
    ]


def catalog_hash(options: Iterable[Option]) -> str:
    """Stable hash of a candidate set, used for reproducible evaluation."""
    h = hashlib.sha256()
    for o in options:
        h.update(f"{o.id}\x00{o.name}\x00{o.description}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def total_chars(options: Iterable[Option]) -> int:
    """Character footprint of a candidate set (name + description), used for
    the flat-vs-hierarchical decision."""
    return sum(len(o.name) + len(o.description) for o in options)
