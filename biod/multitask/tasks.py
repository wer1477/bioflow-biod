"""Task registry for the five biod decisions (multitask v1).

One task = one head + one loss + one renderer + one decode rule. Keep the
names identical to the harness decision points so configs do not drift:

    tool_select | kb_select | context_retention | role_select | relay_select
"""

from __future__ import annotations

from dataclasses import dataclass, field

TASKS = (
    "tool_select",
    "kb_select",
    "context_retention",
    "role_select",
    "relay_select",
    "premise_judge",
    "overclaim_flag",
)

ROLES = ("pi", "searcher", "researcher", "writer", "reviewer")
CONTEXT_LABELS = ("keep", "summarise", "drop")
RELAY_VERDICTS = ("allow", "reject", "escalate")
RELAY_REASONS = (
    "target_owns",
    "wrong_owner",
    "duplicates_work",
    "authority",
    "ambiguous",
    "risk",
)

PREMISE_LABELS = (
    "supports",
    "partially_supports",
    "contradicts",
    "unrelated",
    "insufficient",
)
PREMISE_UNITS = ("hypothesis", "claim", "issue")
OVERCLAIM_FLAGS = (
    "scope",
    "causality",
    "certainty",
    "generalization",
    "temporal",
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    head: str  # pointer | pointwise | group3 | role5 | relay3
    max_len: int
    labels: tuple[str, ...] = ()
    description: str = ""
    # metric names used by the evaluator/report
    metrics: tuple[str, ...] = field(default_factory=tuple)


TASK_SPECS: dict[str, TaskSpec] = {
    "tool_select": TaskSpec(
        name="tool_select",
        head="pointer",
        max_len=5120,
        labels=("NO_TOOL",),
        description="Choose one role-visible capability to load next, or NO_TOOL.",
        metrics=("top1_strict", "no_tool_f1", "macro_top1", "r@5", "mrr", "ece", "flip", "p50_ms"),
    ),
    "kb_select": TaskSpec(
        name="kb_select",
        head="pointwise",
        max_len=4096,
        description="Pointwise relevance over role-visible knowledge entries; empty set is legal.",
        metrics=("hit@1", "r@5", "ndcg@5", "empty_f1", "avg_selected", "p50_ms"),
    ),
    "context_retention": TaskSpec(
        name="context_retention",
        head="group3",
        max_len=2048,
        labels=CONTEXT_LABELS,
        description="Per cold-zone group: keep / summarise / drop.",
        metrics=("accuracy_3", "macro_f1", "per_class_recall", "p50_ms"),
    ),
    "role_select": TaskSpec(
        name="role_select",
        head="role5",
        max_len=2048,
        labels=ROLES,
        description="Route a user message to one of the five team roles.",
        metrics=("accuracy_5", "macro_f1", "pi_recall_cross_role", "misroute_rate", "p50_ms"),
    ),
    "relay_select": TaskSpec(
        name="relay_select",
        head="relay3",
        max_len=2048,
        labels=RELAY_VERDICTS,
        description="Approve, reject, or escalate a role-to-role relay.",
        metrics=("accuracy_3", "macro_f1", "unsafe_allow_rate", "escalate_recall", "p50_ms"),
    ),
    "premise_judge": TaskSpec(
        name="premise_judge",
        head="premise5",
        max_len=4096,
        labels=PREMISE_LABELS,
        description="Judge how an evidence passage relates to a hypothesis, claim, or review-issue premise.",
        metrics=("accuracy_5", "macro_f1", "per_class_recall", "ece", "p50_ms"),
    ),
    "overclaim_flag": TaskSpec(
        name="overclaim_flag",
        head="flag5",
        max_len=4096,
        labels=OVERCLAIM_FLAGS,
        description="Multi-label overclaim detection over a claim given its evidence.",
        metrics=("micro_f1", "macro_f1", "subset_accuracy", "hamming", "p50_ms"),
    ),
}


def normalize_task(task: str) -> str:
    if task in TASK_SPECS:
        return task
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def task_head(task: str) -> str:
    return TASK_SPECS[normalize_task(task)].head


def task_max_len(task: str) -> int:
    return TASK_SPECS[normalize_task(task)].max_len
