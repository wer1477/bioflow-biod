"""Dream-RSI style replay simulator.

Core idea borrowed from *Dream-RSI: Recursive Self-Improvement through Evolving
Worlds* (arXiv:2609.14858): a recorded decision ``(candidates, scores, chosen)``
is a replay world. Alternative *selection policies* can be evaluated by walking
the recorded scores only — no model calls, no re-execution.

What this module can and cannot do
----------------------------------
* CAN replay policies that consume recorded candidate scores: argmax, threshold,
  top-k, margin, per-layer overrides, budget preferences.
* CANNOT replay policies that change the model input (new prompt, new model,
  new option set) — those need fresh forward passes. This mirrors the paper's
  boundary: replay evaluates the policy, not the agent.

Selection is non-regressive by construction: the incumbent policy is always part
of the candidate set (same guarantee as the paper's ``argmax`` over
``{pi_t^0, ..., pi_t^{M-1}}``).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


# --------------------------------------------------------------------------- #
# policy definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Policy:
    """A replayable selection policy over recorded scores."""

    name: str
    tau: float = 0.5          # absolute threshold (kb) / min probability (tool)
    top_k: int = 5            # max selected per step
    margin: float = 0.0       # required gap between best and second best (tool)
    allow_empty: bool = True  # whether returning nothing is allowed
    no_tool_margin: float = 0.0  # NO_TOOL must beat best by this gap to abstain

    def apply(self, record: dict) -> list[str]:
        candidates = record.get("candidates") or []
        if not candidates:
            return []
        scored = [(c.get("id"), float(c.get("score") or 0.0)) for c in candidates]
        scored.sort(key=lambda kv: -kv[1])
        kind = record.get("kind", "tool")

        if kind == "kb":
            chosen = [cid for cid, p in scored if p >= self.tau][: self.top_k]
            if not chosen and not self.allow_empty:
                chosen = [scored[0][0]]
            return chosen

        # tool: single choice, honour NO_TOOL probability as a virtual candidate
        best_id, best_p = scored[0]
        no_tool = float(record.get("noToolProb") or 0.0)
        if no_tool > best_p + self.no_tool_margin:
            return []
        if best_p < self.tau:
            return []
        if self.margin > 0 and len(scored) > 1 and (best_p - scored[1][1]) < self.margin:
            return []
        return [best_id]


DEFAULT_POLICIES: tuple[Policy, ...] = (
    Policy("tool_top1", tau=0.0, top_k=1),
    Policy("tool_margin", tau=0.0, top_k=1, margin=0.05),
    Policy("kb_default", tau=0.5, top_k=5),
    Policy("kb_loose", tau=0.3, top_k=5),
    Policy("kb_strict", tau=0.7, top_k=3),
)


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
@dataclass
class ReplayResult:
    policy: Policy
    steps: int = 0
    empty_steps: int = 0
    selected_total: int = 0
    agreement: float = 0.0       # agreement with recorded choice (diagnostic only)
    hit_rate: float = 0.0        # fraction of steps where a gold/label id was selected
    labeled_steps: int = 0
    objective: float = 0.0

    def to_dict(self) -> dict:
        return {
            "policy": self.policy.name,
            "tau": self.policy.tau,
            "topK": self.policy.top_k,
            "margin": self.policy.margin,
            "noToolMargin": self.policy.no_tool_margin,
            "steps": self.steps,
            "emptySteps": self.empty_steps,
            "selectedTotal": self.selected_total,
            "agreement": round(self.agreement, 4),
            "hitRate": round(self.hit_rate, 4),
            "labeledSteps": self.labeled_steps,
            "objective": round(self.objective, 4),
        }


def _labels(record: dict) -> list[str]:
    """Ground-truth ids for a record, if any.

    Supported keys: ``label`` (id or list of ids), ``labels`` (list of ids).
    """
    raw = record.get("labels")
    if raw is None:
        raw = record.get("label")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]


def replay(records: Sequence[dict], policy: Policy) -> ReplayResult:
    """Evaluate one policy over recorded worlds. Zero model calls."""
    result = ReplayResult(policy=policy)
    hits = 0
    agreed = 0
    for rec in records:
        chosen = policy.apply(rec)
        recorded = set(rec.get("chosen") or [])
        result.steps += 1
        if not chosen:
            result.empty_steps += 1
        result.selected_total += len(chosen)
        if recorded and set(chosen) == recorded:
            agreed += 1
        gold = _labels(rec)
        if gold:
            result.labeled_steps += 1
            if set(gold) & set(chosen):
                hits += 1
    result.agreement = agreed / result.steps if result.steps else 0.0
    result.hit_rate = hits / result.labeled_steps if result.labeled_steps else 0.0
    # objective: prefer hit rate when labels exist, else prefer agreement with
    # the recorded behaviour while penalising over-selection.
    if result.labeled_steps:
        result.objective = result.hit_rate - 0.01 * (result.selected_total / max(1, result.steps))
    else:
        avg_selected = result.selected_total / max(1, result.steps)
        result.objective = result.agreement - 0.02 * avg_selected
    return result


def replay_all(records: Sequence[dict], policies: Iterable[Policy] = DEFAULT_POLICIES) -> list[ReplayResult]:
    return [replay(records, p) for p in policies]


# --------------------------------------------------------------------------- #
# dreaming: offline policy search
# --------------------------------------------------------------------------- #
@dataclass
class DreamResult:
    best: ReplayResult
    incumbent: ReplayResult
    grid: list[ReplayResult] = field(default_factory=list)
    improved: bool = False

    def to_dict(self) -> dict:
        return {
            "improved": self.improved,
            "incumbent": self.incumbent.to_dict(),
            "best": self.best.to_dict(),
            "grid": [r.to_dict() for r in self.grid],
        }


def policy_grid(
    taus: Sequence[float] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    top_ks: Sequence[int] = (1, 3, 5),
    margins: Sequence[float] = (0.0, 0.05, 0.1),
    allow_empty: Sequence[bool] = (True,),
    no_tool_margins: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5),
) -> list[Policy]:
    """Cartesian policy grid used by the dreaming step."""
    grid = []
    for tau, k, margin, empty, nt_margin in itertools.product(taus, top_ks, margins, allow_empty, no_tool_margins):
        grid.append(
            Policy(
                f"tau{tau}_k{k}_m{margin}_e{int(empty)}_nt{nt_margin}",
                tau=tau,
                top_k=k,
                margin=margin,
                allow_empty=empty,
                no_tool_margin=nt_margin,
            )
        )
    return grid


def dream(
    records: Sequence[dict],
    incumbent: Policy | None = None,
    grid: Sequence[Policy] | None = None,
) -> DreamResult:
    """Evaluate a policy grid on recorded worlds and pick the best.

    The incumbent is included in the candidate set, so the selected policy is
    never worse than the current one on the recorded worlds (the paper's
    non-regression guarantee, restricted to replay).
    """
    incumbent = incumbent or Policy("incumbent", tau=0.5, top_k=5)
    grid = list(grid) if grid is not None else policy_grid()
    evaluated = [replay(records, p) for p in [incumbent, *grid]]
    inc_res = evaluated[0]
    best = max(evaluated, key=lambda r: r.objective)
    return DreamResult(best=best, incumbent=inc_res, grid=evaluated[1:], improved=best.objective > inc_res.objective)


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def load_records(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_replay_results(path: str, result: DreamResult) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)
