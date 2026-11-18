"""Decision engine: hierarchical or flat selection over the two catalogs.

Two decision semantics (matching ``docs/筛选决策模型设计.md``):

* ``tool``  -> N-select-one per layer (listwise letter readout, optional NO_TOOL)
* ``kb``    -> absolute pointwise scores + threshold, variable hit count

Every step is logged as a :class:`DecisionRecord` so that the Dream-RSI style
replay simulator can re-evaluate alternative policies **without any model
calls**.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .backend import Backend
from .catalog import (
    Domain,
    Option,
    catalog_hash,
    domain_as_options,
    load_knowledge_base,
    load_tool_library,
    total_chars,
)
from .readout import ChoiceResult, letter_choice, pointwise_scores
from .render import DEFAULT_SCORE_INSTRUCTION, DEFAULT_TOOL_POINTWISE_INSTRUCTION

NO_TOOL_ID = "__no_tool__"


@dataclass
class Step:
    """One layer of one decision, plus everything needed to replay it."""

    layer: str
    candidates: list[Option]
    scores: dict[str, float]          # id -> probability / absolute score
    chosen: list[str]
    policy: str
    catalog_hash: str = ""
    fallback: bool = False
    no_tool_prob: float = 0.0

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "policy": self.policy,
            "catalogHash": self.catalog_hash or catalog_hash(self.candidates),
            "fallback": self.fallback,
            "noToolProb": self.no_tool_prob,
            "candidates": [
                {**o.to_dict(), "score": self.scores.get(o.id)} for o in self.candidates
            ],
            "chosen": list(self.chosen),
        }


@dataclass
class Decision:
    kind: str                          # tool | kb
    mode: str                          # hierarchical | flat
    state: str
    instruction: str
    steps: list[Step] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    backend: str = "mock"

    def to_dict(self) -> dict:
        return {
            "decisionId": self.decision_id,
            "kind": self.kind,
            "mode": self.mode,
            "backend": self.backend,
            "instruction": self.instruction,
            "latencyMs": round(self.latency_ms, 1),
            "selected": self.selected,
            "steps": [s.to_dict() for s in self.steps],
        }

    def records(self) -> list[dict]:
        """Flatten into one replay record per step (state duplicated)."""
        out = []
        for step in self.steps:
            out.append(
                {
                    "decisionId": self.decision_id,
                    "kind": self.kind,
                    "layer": step.layer,
                    "state": self.state,
                    "instruction": self.instruction,
                    "policy": step.policy,
                    "backend": self.backend,
                    "catalogHash": step.catalog_hash or catalog_hash(step.candidates),
                    "noToolProb": step.no_tool_prob,
                    "fallback": step.fallback,
                    "candidates": [
                        {**o.to_dict(), "score": step.scores.get(o.id)} for o in step.candidates
                    ],
                    "chosen": list(step.chosen),
                }
            )
        return out


class DecisionEngine:
    """Zero-training decision engine over bioagent-dev's catalogs."""

    def __init__(
        self,
        backend: Backend,
        tool_root: str | None = None,
        kb_root: str | None = None,
        flat_char_limit: int = 30_000,
        tau: float = 0.5,
        max_selected: int = 5,
        no_tool_margin: float = 0.0,
    ):
        self.backend = backend
        self.flat_char_limit = flat_char_limit
        self.tau = tau
        self.max_selected = max_selected
        self.no_tool_margin = no_tool_margin
        self._tool_domains: list[Domain] | None = None
        self._kb_domains: list[Domain] | None = None
        if tool_root:
            self._tool_domains = load_tool_library(tool_root)
        if kb_root:
            self._kb_domains = load_knowledge_base(kb_root)

    # -- catalog accessors -------------------------------------------------- #
    @property
    def tool_domains(self) -> list[Domain]:
        return self._tool_domains or []

    @property
    def kb_domains(self) -> list[Domain]:
        return self._kb_domains or []

    def all_tool_options(self) -> list[Option]:
        out: list[Option] = []
        for d in self.tool_domains:
            out.extend(d.options)
        return out

    def all_kb_options(self) -> list[Option]:
        out: list[Option] = []
        for d in self.kb_domains:
            out.extend(d.options)
        return out

    # -- tool library ------------------------------------------------------- #
    def decide_tool(
        self,
        state: str,
        instruction: str = "Select exactly one capability to load next. Choose NO_TOOL if none applies.",
        mode: str = "auto",
        domain: str | None = None,
        min_prob: float = 0.0,
        pointwise_instruction: str | None = None,
    ) -> Decision:
        started = time.time()
        decision = Decision(kind="tool", mode=mode, state=state, instruction=instruction, backend=self.backend.name)

        if domain:
            target = next((d for d in self.tool_domains if d.name == domain), None)
            if target is None:
                raise ValueError(f"unknown tool domain: {domain}")
            decision.mode = "hierarchical"
            self._tool_l2(decision, target)
        else:
            flat = self.all_tool_options()
            use_flat = mode == "flat" or (mode == "auto" and total_chars(flat) <= self.flat_char_limit and len(flat) <= 26)
            if use_flat and flat:
                decision.mode = "flat"
                self._tool_flat(decision, flat, pointwise_instruction)
            elif mode == "hier_pointwise":
                # Zero-training alternative readout for Base models: binary
                # pointwise judgement at both levels instead of listwise letters.
                decision.mode = "hierarchical"
                domains = self.tool_domains
                step = self._pointwise_choice_step(
                    state,
                    domain_as_options(domains, "tool_l1"),
                    layer="tool_l1",
                    instruction=pointwise_instruction or DEFAULT_TOOL_POINTWISE_INSTRUCTION,
                )
                decision.steps.append(step)
                picked = step.chosen[0] if step.chosen else None
                if picked:
                    target = next((d for d in domains if d.name == picked), None)
                    if target is not None:
                        self._tool_l2_pointwise(decision, target, pointwise_instruction)
            else:
                decision.mode = "hierarchical"
                domains = self.tool_domains
                step = self._choice_step(state, instruction, domain_as_options(domains, "tool_l1"), layer="tool_l1")
                decision.steps.append(step)
                picked = step.chosen[0] if step.chosen else None
                if picked:
                    target = next((d for d in domains if d.name == picked), None)
                    if target is not None:
                        self._tool_l2(decision, target)

        decision.selected = decision.steps[-1].chosen if decision.steps else []
        decision.latency_ms = (time.time() - started) * 1000
        return decision

    def _tool_l2(self, decision: Decision, domain: Domain) -> None:
        if not domain.options:
            decision.steps.append(Step("tool_l2", [], {}, [], "empty"))
            return
        step = self._choice_step(decision.state, decision.instruction, domain.options, layer="tool_l2")
        decision.steps.append(step)

    def _tool_flat(
        self,
        decision: Decision,
        options: Sequence[Option],
        pointwise_instruction: str | None = None,
    ) -> None:
        if len(options) <= 26:
            decision.steps.append(self._choice_step(decision.state, decision.instruction, options, layer="tool_l2"))
            return
        # > 26 options: absolute pointwise scores, argmax (no letter limit)
        scores = pointwise_scores(
            self.backend,
            decision.state,
            options,
            instruction=pointwise_instruction or DEFAULT_TOOL_POINTWISE_INSTRUCTION,
        )
        mapping = {o.id: float(p) for o, p in zip(options, scores)}
        chosen = [max(mapping, key=mapping.get)] if mapping else []
        decision.steps.append(
            Step("tool_l2", list(options), mapping, chosen, "pointwise_argmax", catalog_hash(options))
        )

    def _pointwise_choice_step(
        self,
        state: str,
        options: Sequence[Option],
        layer: str,
        instruction: str = DEFAULT_TOOL_POINTWISE_INSTRUCTION,
    ) -> Step:
        scores = pointwise_scores(self.backend, state, options, instruction=instruction)
        mapping = {o.id: float(p) for o, p in zip(options, scores)}
        chosen = [max(mapping, key=mapping.get)] if mapping else []
        return Step(
            layer=layer,
            candidates=list(options),
            scores=mapping,
            chosen=chosen,
            policy="pointwise_argmax",
            catalog_hash=catalog_hash(options),
        )

    def _tool_l2_pointwise(
        self,
        decision: Decision,
        domain: Domain,
        pointwise_instruction: str | None = None,
    ) -> None:
        if not domain.options:
            decision.steps.append(Step("tool_l2", [], {}, [], "empty"))
            return
        decision.steps.append(
            self._pointwise_choice_step(
                decision.state,
                domain.options,
                layer="tool_l2",
                instruction=pointwise_instruction or DEFAULT_TOOL_POINTWISE_INSTRUCTION,
            )
        )

    def _choice_step(self, state: str, instruction: str, options: Sequence[Option], layer: str) -> Step:
        result: ChoiceResult = letter_choice(self.backend, state, options, instruction=instruction)
        chosen = result.selected_id(no_tool_margin=self.no_tool_margin)
        return Step(
            layer=layer,
            candidates=list(options),
            scores=dict(result.probs),
            chosen=[chosen] if chosen else [],
            policy="letter_choice",
            catalog_hash=catalog_hash(options),
            no_tool_prob=result.no_tool_prob,
        )

    # -- knowledge base ----------------------------------------------------- #
    def decide_kb(
        self,
        state: str,
        instruction: str = "Judge whether the candidate knowledge entry is useful for the current need.",
        mode: str = "auto",
        tau: float | None = None,
        max_selected: int | None = None,
        domain: str | None = None,
    ) -> Decision:
        started = time.time()
        tau = self.tau if tau is None else tau
        max_selected = self.max_selected if max_selected is None else max_selected
        decision = Decision(kind="kb", mode=mode, state=state, instruction=instruction, backend=self.backend.name)

        if domain:
            target = next((d for d in self.kb_domains if d.name == domain), None)
            if target is None:
                raise ValueError(f"unknown kb domain: {domain}")
            decision.mode = "hierarchical"
            self._kb_l2(decision, target, instruction, tau, max_selected)
        else:
            flat = self.all_kb_options()
            use_flat = mode == "flat" or (mode == "auto" and total_chars(flat) <= self.flat_char_limit)
            if use_flat and flat:
                decision.mode = "flat"
                scores = pointwise_scores(self.backend, state, flat, instruction=instruction)
                mapping = {o.id: float(p) for o, p in zip(flat, scores)}
                chosen = [oid for oid, p in sorted(mapping.items(), key=lambda kv: -kv[1]) if p >= tau][:max_selected]
                decision.steps.append(Step("kb_l2", list(flat), mapping, chosen, "pointwise_threshold", catalog_hash(flat)))
            else:
                decision.mode = "hierarchical"
                domain_opts = domain_as_options(self.kb_domains, "kb_l1")
                d_scores = pointwise_scores(self.backend, state, domain_opts, instruction=instruction)
                d_map = {o.id: float(p) for o, p in zip(domain_opts, d_scores)}
                picked = [oid for oid, p in sorted(d_map.items(), key=lambda kv: -kv[1]) if p >= tau]
                fallback = False
                if not picked and d_map:
                    picked = [max(d_map, key=d_map.get)]
                    fallback = True
                decision.steps.append(
                    Step("kb_l1", list(domain_opts), d_map, picked, "pointwise_threshold", catalog_hash(domain_opts), fallback)
                )
                for name in picked:
                    target = next((d for d in self.kb_domains if d.name == name), None)
                    if target is not None:
                        self._kb_l2(decision, target, instruction, tau, max_selected)

        decision.selected = decision.steps[-1].chosen if decision.steps else []
        decision.latency_ms = (time.time() - started) * 1000
        return decision

    def _kb_l2(self, decision: Decision, domain: Domain, instruction: str, tau: float, max_selected: int) -> None:
        if not domain.options:
            decision.steps.append(Step("kb_l2", [], {}, [], "empty"))
            return
        scores = pointwise_scores(self.backend, decision.state, domain.options, instruction=instruction)
        mapping = {o.id: float(p) for o, p in zip(domain.options, scores)}
        chosen = [oid for oid, p in sorted(mapping.items(), key=lambda kv: -kv[1]) if p >= tau][:max_selected]
        decision.steps.append(
            Step("kb_l2", list(domain.options), mapping, chosen, "pointwise_threshold", catalog_hash(domain.options))
        )


# --------------------------------------------------------------------------- #
# record persistence
# --------------------------------------------------------------------------- #
def write_records(path: str, records: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_records(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
