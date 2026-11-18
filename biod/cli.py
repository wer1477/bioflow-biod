"""Command line interface for the zero-training Biod decision model.

Examples
--------
List the catalogs::

    python -m biod catalog --kind tool
    python -m biod catalog --kind kb --json

Decide with the deterministic mock backend (no GPU, no weights)::

    python -m biod decide --kind tool --mock --state "需要对单细胞 RNA-seq 做质控和聚类"

Decide with a real base model::

    HF_ENDPOINT=https://hf-mirror.com python -m biod decide --kind kb \\
        --model Qwen/Qwen3.5-2B-Base --state "..." --tau 0.5 --record runs/kb.jsonl

Replay / dream over recorded decisions (zero model calls)::

    python -m biod replay --records runs/kb.jsonl --out artifacts/replay.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .backend import load_backend
from .catalog import load_knowledge_base, load_tool_library, total_chars
from .decide import DecisionEngine, read_records, write_records
from .evaluate import evaluate, format_table, load_eval_set
from .replay import Policy, dream, replay_all

DEFAULT_AGENT_ROOT = os.environ.get(
    "BIOD_AGENT_ROOT",
    "/path/to/bioflow/Agent/.mirror/bioflow-dev/agent-client",
)


def _engine(args) -> DecisionEngine:
    root = args.root
    tool_root = os.path.join(root, "toollibrary") if root else None
    kb_root = os.path.join(root, "knowledgebase") if root else None
    backend = load_backend(
        getattr(args, "model", None),
        device=getattr(args, "device", "cuda"),
        dtype=getattr(args, "dtype", "bfloat16"),
        mock=getattr(args, "mock", False),
    )
    return DecisionEngine(
        backend,
        tool_root=tool_root,
        kb_root=kb_root,
        tau=getattr(args, "tau", 0.5),
        max_selected=getattr(args, "max_selected", 5),
        no_tool_margin=getattr(args, "no_tool_margin", 0.0),
    )


def cmd_catalog(args) -> int:
    root = args.root
    if args.kind == "tool":
        domains = load_tool_library(os.path.join(root, "toollibrary"))
    else:
        domains = load_knowledge_base(os.path.join(root, "knowledgebase"))
    payload = {
        "kind": args.kind,
        "root": root,
        "domainCount": len(domains),
        "entryCount": sum(len(d.options) for d in domains),
        "charFootprint": total_chars([o for d in domains for o in d.options]),
        "domains": [d.to_dict() for d in domains],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"[{args.kind}] {payload['domainCount']} domains, {payload['entryCount']} entries, "
              f"{payload['charFootprint']} chars")
        for d in domains:
            print(f"  - {d.name} ({len(d.options)}): {d.description[:60]}")
    return 0


def _read_state(args) -> str:
    if args.state_file:
        with open(args.state_file, encoding="utf-8") as fh:
            return fh.read()
    if args.state:
        return args.state
    return ""


def cmd_decide(args) -> int:
    engine = _engine(args)
    state = _read_state(args)
    if not state.strip():
        print("error: provide --state or --state-file", file=sys.stderr)
        return 2
    instruction = args.instruction
    if args.kind == "tool":
        decision = engine.decide_tool(state, mode=args.mode, domain=args.domain, instruction=instruction or
                                      "Select exactly one capability to load next. Choose NO_TOOL if none applies.")
    else:
        decision = engine.decide_kb(state, mode=args.mode, domain=args.domain, tau=args.tau,
                                    max_selected=args.max_selected, instruction=instruction or
                                    "Judge whether the candidate knowledge entry is useful for the current need.")

    out = decision.to_dict()
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.record:
        n = write_records(args.record, decision.records())
        print(f"\n[recorded {n} step(s) -> {args.record}]", file=sys.stderr)
    return 0


def cmd_replay(args) -> int:
    records = read_records(args.records)
    if not records:
        print("error: no records", file=sys.stderr)
        return 2
    kinds = {r.get("kind") for r in records}
    is_kb = "kb" in kinds
    if args.taus:
        policies = [Policy(f"tau{t}", tau=t, top_k=args.top_k) for t in _floats(args.taus)]
        results = replay_all(records, policies)
        best = max(results, key=lambda r: r.objective)
        payload = {"records": len(records), "results": [r.to_dict() for r in results],
                   "best": best.to_dict()}
    else:
        incumbent_tau = args.incumbent_tau if args.incumbent_tau is not None else (0.5 if is_kb else 0.0)
        top_k = args.top_k if not is_kb else max(args.top_k, 1)
        incumbent = Policy("incumbent", tau=incumbent_tau, top_k=top_k)
        result = dream(records, incumbent=incumbent)
        payload = {"records": len(records), **result.to_dict()}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[wrote {args.out}]", file=sys.stderr)
    print(text)
    return 0


def cmd_demo(args) -> int:
    """A self-contained zero-training demo over the real catalogs."""
    args.kind = "tool"
    engine = _engine(args)
    state = ("我需要分析一批单细胞 RNA-seq 数据：先做质控和标准化，然后聚类并注释细胞类型，"
             "最后做差异表达分析。")
    tool_dec = engine.decide_tool(state, mode=args.mode)
    kb_dec = engine.decide_kb(state, mode=args.mode, tau=args.tau)
    payload = {
        "state": state,
        "tool": tool_dec.to_dict(),
        "kb": kb_dec.to_dict(),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[wrote {args.out}]", file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_evaluate(args) -> int:
    engine = _engine(args)
    items = load_eval_set(args.eval_set)
    result = evaluate(engine, items, mode=args.mode, tau=args.tau, top_k=args.top_k,
                      records_path=args.dump_records,
                      tool_instruction=args.tool_instruction,
                      kb_instruction=args.kb_instruction,
                      tool_mode=args.tool_mode,
                      kb_mode=args.kb_mode,
                      tool_pointwise_instruction=args.tool_pointwise_instruction)
    text = format_table(result)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"[wrote {args.out}]", file=sys.stderr)
    print(text)
    return 0


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _add_common(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """Common flags accepted both before and after the subcommand.

    On subparsers we use ``argparse.SUPPRESS`` so that a subcommand-level default
    does not clobber a value the user passed *before* the subcommand.
    """

    def _kw(default):
        return {"default": argparse.SUPPRESS} if suppress else {"default": default}

    parser.add_argument("--root", help="agent-client root containing toollibrary/ and knowledgebase/",
                        **_kw(DEFAULT_AGENT_ROOT))
    parser.add_argument("--model", help="HF model id or local path; omit with --mock", **_kw(None))
    parser.add_argument("--mock", action="store_true", help="use deterministic mock backend (no model)", **_kw(False))
    parser.add_argument("--device", **_kw("cuda"))
    parser.add_argument("--dtype", **_kw("bfloat16"))
    parser.add_argument("--no-tool-margin", type=float, **_kw(0.0),
                        help="NO_TOOL must beat the best candidate by this margin to abstain "
                             "(positive = fewer empty selections)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biod", description="Zero-training Biod decision model")
    _add_common(p)
    sub_common = argparse.ArgumentParser(add_help=False)
    _add_common(sub_common, suppress=True)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("catalog", parents=[sub_common], help="inspect catalogs")
    c.add_argument("--kind", choices=["tool", "kb"], default="tool")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_catalog)

    d = sub.add_parser("decide", parents=[sub_common], help="run one decision")
    d.add_argument("--kind", choices=["tool", "kb"], required=True)
    d.add_argument("--state", default=None)
    d.add_argument("--state-file", default=None)
    d.add_argument("--instruction", default=None)
    d.add_argument("--mode", choices=["auto", "flat", "hierarchical"], default="auto")
    d.add_argument("--domain", default=None)
    d.add_argument("--tau", type=float, default=0.5)
    d.add_argument("--max-selected", type=int, default=5)
    d.add_argument("--record", default=None, help="append step records to this jsonl")
    d.set_defaults(func=cmd_decide)

    r = sub.add_parser("replay", parents=[sub_common], help="replay/dream over recorded decisions (zero model calls)")
    r.add_argument("--records", required=True)
    r.add_argument("--taus", default=None, help="comma separated thresholds; omit to run the full dream grid")
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--incumbent-tau", type=float, default=None, help="default: 0.5 for kb, 0.0 for tool")
    r.add_argument("--out", default=None)
    r.set_defaults(func=cmd_replay)

    e = sub.add_parser("evaluate", parents=[sub_common], help="evaluate on a JSONL eval set")
    e.add_argument("--eval-set", required=True)
    e.add_argument("--mode", choices=["auto", "flat", "hierarchical"], default="auto")
    e.add_argument("--tau", type=float, default=0.5)
    e.add_argument("--top-k", type=int, default=5)
    e.add_argument("--out", default=None)
    e.add_argument("--dump-records", default=None,
                   help="write replayable decision records (with labels) to this jsonl")
    e.add_argument("--tool-instruction", default=None, help="override tool choice instruction")
    e.add_argument("--kb-instruction", default=None, help="override kb pointwise instruction")
    e.add_argument("--tool-mode", choices=["auto", "flat", "hierarchical", "hier_pointwise"], default=None,
                   help="override mode for tool items only")
    e.add_argument("--tool-pointwise-instruction", default=None,
                   help="override instruction for pointwise tool scoring")
    e.add_argument("--kb-mode", choices=["auto", "flat", "hierarchical"], default=None,
                   help="override mode for kb items only")
    e.set_defaults(func=cmd_evaluate)

    m = sub.add_parser("demo", parents=[sub_common], help="run a tool+kb demo on the real catalogs")
    m.add_argument("--mode", choices=["auto", "flat", "hierarchical"], default="auto")
    m.add_argument("--tau", type=float, default=0.5)
    m.add_argument("--out", default=None)
    m.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # subcommand-level --root/--model are SUPPRESSed so the top-level value wins;
    # fall back to the env-derived default when neither was given.
    for name, default in (("root", DEFAULT_AGENT_ROOT), ("model", None), ("mock", False),
                          ("device", "cuda"), ("dtype", "bfloat16")):
        if not hasattr(args, name) or getattr(args, name) is None:
            if name == "root":
                setattr(args, name, default)
    if not getattr(args, "root", None):
        print("error: --root is required", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
