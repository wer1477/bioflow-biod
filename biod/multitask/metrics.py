"""Metrics for the five decisions. Pure Python (no numpy dependency)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from .tasks import OVERCLAIM_FLAGS, PREMISE_LABELS


def _div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def accuracy(gold: Sequence, pred: Sequence) -> float:
    return _div(sum(1 for g, p in zip(gold, pred) if g == p), len(gold))


def confusion(gold: Sequence, pred: Sequence, labels: Sequence) -> dict:
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            matrix[idx[g]][idx[p]] += 1
    return {"labels": list(labels), "matrix": matrix}


def macro_f1(gold: Sequence, pred: Sequence, labels: Sequence) -> float:
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = _div(tp, tp + fp)
        recall = _div(tp, tp + fn)
        f1s.append(_div(2 * precision * recall, precision + recall))
    return _div(sum(f1s), len(f1s))


def per_class_recall(gold: Sequence, pred: Sequence, labels: Sequence) -> dict:
    return {label: _div(sum(1 for g, p in zip(gold, pred) if g == label and p == label),
                        sum(1 for g in gold if g == label)) for label in labels}


# --- tool -------------------------------------------------------------------
def tool_metrics(records: Iterable[dict]) -> dict:
    """Records: gold_id, chosen_id, ranked_ids (best first), probs (id->p), no_tool_prob, confidence."""
    rows = list(records)
    if not rows:
        return {}
    gold = [r["gold_id"] for r in rows]
    chosen = [r["chosen_id"] for r in rows]
    strict = _div(sum(1 for g, c in zip(gold, chosen) if g == c), len(rows))
    # forced top-1 excludes NO_TOOL gold rows
    forced_rows = [(g, r) for g, r in zip(gold, rows) if g != "NO_TOOL"]
    forced = _div(sum(1 for g, r in forced_rows if r.get("ranked_ids") and r["ranked_ids"][0] == g), len(forced_rows))
    tp = sum(1 for g, c in zip(gold, chosen) if g == "NO_TOOL" and c == "NO_TOOL")
    fp = sum(1 for g, c in zip(gold, chosen) if g != "NO_TOOL" and c == "NO_TOOL")
    fn = sum(1 for g, c in zip(gold, chosen) if g == "NO_TOOL" and c != "NO_TOOL")
    p = _div(tp, tp + fp); r = _div(tp, tp + fn)
    no_tool_f1 = _div(2 * p * r, p + r)
    # macro top-1 over gold tools
    by_gold = defaultdict(lambda: [0, 0])
    for g, c in zip(gold, chosen):
        by_gold[g][1] += 1
        by_gold[g][0] += int(g == c)
    macro = _div(sum(_div(a, b) for a, b in by_gold.values()), len(by_gold))
    rr, r5 = [], []
    for g, row in zip(gold, rows):
        ranked = row["ranked_ids"]
        if g == "NO_TOOL":
            continue
        rank = ranked.index(g) + 1 if g in ranked else None
        rr.append(1.0 / rank if rank else 0.0)
        r5.append(1.0 if rank and rank <= 5 else 0.0)
    ece = 0.0
    bins = 10
    bucket = [[0, 0.0, 0.0] for _ in range(bins)]  # n, conf, acc
    for g, c, conf in zip(gold, chosen, [r["confidence"] for r in rows]):
        b = min(bins - 1, int(conf * bins))
        bucket[b][0] += 1
        bucket[b][1] += conf
        bucket[b][2] += float(g == c)
    for n, conf_sum, acc_sum in bucket:
        if n:
            ece += abs(acc_sum / n - conf_sum / n) * n / len(rows)
    return {
        "n": len(rows),
        "top1_strict": strict,
        "top1_forced": forced,
        "no_tool_f1": no_tool_f1,
        "macro_top1": macro,
        "r@5": _div(sum(r5), len(r5)),
        "mrr": _div(sum(rr), len(rr)),
        "ece": ece,
    }


# --- kb ---------------------------------------------------------------------
def kb_metrics(records: Iterable[dict]) -> dict:
    """Records: gold_ids (grade 2), selected_ids, scores (id->p)."""
    rows = list(records)
    if not rows:
        return {}
    hit1, r5, ndcg, empty_tp, empty_fp, empty_fn, avg_sel = [], [], [], 0, 0, 0, []
    empty_gold = 0
    for row in rows:
        scores = row["scores"]
        ranked = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        gold = set(row["gold_ids"])
        selected = list(row["selected_ids"])
        if gold:
            top = ranked[0] if ranked else None
            hit1.append(1.0 if top in gold else 0.0)
            r5.append(_div(len(gold & set(ranked[:5])), len(gold)))
            dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:5]) if cid in gold)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 5)))
            ndcg.append(_div(dcg, ideal))
        else:
            empty_gold += 1
        pred_empty = not selected
        if pred_empty and not gold:
            empty_tp += 1
        elif pred_empty and gold:
            empty_fn += 1
        elif not pred_empty and not gold:
            empty_fp += 1
        avg_sel.append(len(selected))
    ep = _div(empty_tp, empty_tp + empty_fp); er = _div(empty_tp, empty_tp + empty_fn)
    return {
        "n": len(rows),
        "hit@1": _div(sum(hit1), len(hit1)),
        "r@5": _div(sum(r5), len(r5)),
        "ndcg@5": _div(sum(ndcg), len(ndcg)),
        "empty_f1": _div(2 * ep * er, ep + er),
        "avg_selected": _div(sum(avg_sel), len(avg_sel)),
        "n_empty_gold": empty_gold,
    }


# --- context ----------------------------------------------------------------
def context_metrics(records: Iterable[dict]) -> dict:
    rows = list(records)
    labels = ["keep", "summarise", "drop"]
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    return {
        "n": len(rows),
        "accuracy_3": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, labels),
        "per_class_recall": per_class_recall(gold, pred, labels),
        "confusion": confusion(gold, pred, labels),
    }


# --- role -------------------------------------------------------------------
def role_metrics(records: Iterable[dict]) -> dict:
    rows = list(records)
    labels = ["pi", "searcher", "researcher", "writer", "reviewer"]
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    # pi recall on cross-role rows needs the generator
    cross_rows = [r for r in rows if str(r.get("generator", "")).startswith("pi_cross")]
    pi_recall = _div(sum(1 for r in cross_rows if r["pred"] == "pi"), len(cross_rows))
    non_pi = [(r["gold"], r["pred"]) for r in rows if r["gold"] != "pi"]
    misroute = _div(sum(1 for g, p in non_pi if p != g), len(non_pi))
    return {
        "n": len(rows),
        "accuracy_5": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, labels),
        "pi_recall_cross_role": pi_recall,
        "misroute_rate": misroute,
        "per_class_recall": per_class_recall(gold, pred, labels),
        "confusion": confusion(gold, pred, labels),
    }


def r_is_cross(_row: dict) -> bool:  # kept for API stability
    return False


# --- relay ------------------------------------------------------------------
def relay_metrics(records: Iterable[dict]) -> dict:
    rows = list(records)
    labels = ["allow", "reject", "escalate"]
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    unsafe = sum(1 for g, p in zip(gold, pred) if g in {"reject", "escalate"} and p == "allow")
    esc = [(g, p) for g, p in zip(gold, pred) if g == "escalate"]
    return {
        "n": len(rows),
        "accuracy_3": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, labels),
        "unsafe_allow_rate": _div(unsafe, len(rows)),
        "escalate_recall": _div(sum(1 for g, p in esc if p == "escalate"), len(esc)),
        "confusion": confusion(gold, pred, labels),
        "per_class_recall": per_class_recall(gold, pred, labels),
    }


# --- role heads -------------------------------------------------------------
def _ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> float:
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for c, ok in zip(confidences, correct):
        bins[min(n_bins - 1, int(float(c) * n_bins))].append((float(c), float(ok)))
    total = len(confidences)
    if not total:
        return 0.0
    err = 0.0
    for idx in range(n_bins):
        items = bins.get(idx) or []
        if not items:
            continue
        avg_conf = sum(x[0] for x in items) / len(items)
        acc = sum(x[1] for x in items) / len(items)
        err += len(items) / total * abs(avg_conf - acc)
    return err


def premise_metrics(records: Iterable[dict]) -> dict:
    """Records: gold (relation), pred (relation), confidence."""
    rows = list(records)
    if not rows:
        return {}
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    conf = [float(r.get("confidence", 1.0)) for r in rows]
    correct = [1.0 if g == p else 0.0 for g, p in zip(gold, pred)]
    return {
        "accuracy_5": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, PREMISE_LABELS),
        "per_class_recall": per_class_recall(gold, pred, PREMISE_LABELS),
        "ece": _ece(conf, correct),
        "n": len(rows),
    }


def flag_metrics(records: Iterable[dict]) -> dict:
    """Records: gold_flags (list), pred_flags (list)."""
    rows = list(records)
    if not rows:
        return {}
    tp = fp = fn = 0
    per: dict[str, list[int]] = {f: [0, 0, 0] for f in OVERCLAIM_FLAGS}
    exact = 0
    hamming = 0.0
    for r in rows:
        g = set(r.get("gold_flags") or [])
        p = set(r.get("pred_flags") or [])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        exact += 1 if g == p else 0
        hamming += len(g ^ p) / max(1, len(OVERCLAIM_FLAGS))
        for f in OVERCLAIM_FLAGS:
            if f in g and f in p:
                per[f][0] += 1
            elif f in p:
                per[f][1] += 1
            elif f in g:
                per[f][2] += 1
    micro_p = _div(tp, tp + fp)
    micro_r = _div(tp, tp + fn)
    f1s = []
    for f, (a, b, c) in per.items():
        p = _div(a, a + b)
        r = _div(a, a + c)
        f1s.append(_div(2 * p * r, p + r))
    return {
        "micro_f1": _div(2 * micro_p * micro_r, micro_p + micro_r),
        "macro_f1": _div(sum(f1s), len(f1s)),
        "subset_accuracy": _div(exact, len(rows)),
        "hamming": _div(hamming, len(rows)),
        "per_label_f1": {f: f1s[i] for i, f in enumerate(per)},
        "n": len(rows),
    }
