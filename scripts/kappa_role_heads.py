#!/usr/bin/env python
"""Cohen's kappa between independent annotators on the role-head probe.

Reads the blinded probe items, the generator labels and the relabel files, then
reports agreement for premise_judge (multi-class) and overclaim_flag (per-label
binary + mean).
"""
from __future__ import annotations

import json
import os
import sys

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "Datasets/v3/role_heads")
FLAGS = ("scope", "causality", "certainty", "generalization", "temporal")


def load(path: str) -> dict[str, dict]:
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["id"])] = rec
    return out


def kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for label in labels:
        pa = sum(1 for a, _ in pairs if a == label) / n
        pb = sum(1 for _, b in pairs if b == label) / n
        pe += pa * pb
    return 0.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def main() -> int:
    gen_p = load(os.path.join(BASE, "raw/premise_judge/kappa-gen.jsonl"))
    gen_o = load(os.path.join(BASE, "raw/overclaim_flag/kappa-gen.jsonl"))
    a_p = load(os.path.join(BASE, "kappa/label-A-premise_judge.jsonl"))
    b_p = load(os.path.join(BASE, "kappa/label-B-premise_judge.jsonl"))
    a_o = load(os.path.join(BASE, "kappa/label-A-overclaim_flag.jsonl"))
    b_o = load(os.path.join(BASE, "kappa/label-B-overclaim_flag.jsonl"))

    rel = ("supports", "partially_supports", "contradicts", "unrelated", "insufficient")
    ids = sorted(set(gen_p) & set(a_p) & set(b_p))
    pairs_ab = [(a_p[i]["relation"], b_p[i]["relation"]) for i in ids]
    pairs_ag = [(a_p[i]["relation"], gen_p[i]["label"]["relation"]) for i in ids]
    pairs_bg = [(b_p[i]["relation"], gen_p[i]["label"]["relation"]) for i in ids]
    print(f"premise_judge: n={len(ids)}")
    print(f"  kappa A-B = {kappa(pairs_ab, list(rel)):.3f}")
    print(f"  kappa A-gen = {kappa(pairs_ag, list(rel)):.3f}")
    print(f"  kappa B-gen = {kappa(pairs_bg, list(rel)):.3f}")

    ids_o = sorted(set(gen_o) & set(a_o) & set(b_o))
    gen_flags = {i: set((gen_o[i].get("label") or {}).get("flags") or []) for i in gen_o}
    print(f"\noverclaim_flag: n={len(ids_o)}")
    for name, left, right in (("A-B", a_o, b_o), ("A-gen", a_o, gen_flags), ("B-gen", b_o, gen_flags)):
        ks, micro_pairs = [], []
        for flag in FLAGS:
            pairs = [
                ("1" if flag in set(left[i].get("flags") or []) else "0",
                 "1" if flag in set(right[i].get("flags") if isinstance(right[i], dict) and "flags" in right[i] else right[i]) else "0")
                for i in ids_o
            ]
            ks.append(kappa(pairs, ["0", "1"]))
            micro_pairs.extend(pairs)
        print(f"  kappa {name}: mean-per-label={sum(ks)/len(ks):.3f} micro={kappa(micro_pairs, ['0','1']):.3f} "
              f"per-label={[round(k,2) for k in ks]}")

    ok = (kappa(pairs_ab, list(rel)) >= 0.70)
    print(f"\nGATE premise A-B >= 0.70: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
