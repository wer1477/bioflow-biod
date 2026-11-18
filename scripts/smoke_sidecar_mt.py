#!/usr/bin/env python3
"""Smoke-test the biod-mt sidecar over HTTP (assumes it is already running)."""
from __future__ import annotations
import json, sys, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8791"


def call(path, payload=None):
    url = BASE + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main() -> int:
    health = call("/v1/health")
    version = call("/v1/version")
    print("health", json.dumps(health, ensure_ascii=False))
    print("version", json.dumps(version, ensure_ascii=False))
    tool = call("/v1/decide", {
        "requestId": "smoke-tool", "kind": "tool", "role": "researcher",
        "state": "我有 10x 单细胞 counts 矩阵，想先做 QC、聚类和 UMAP，下一步最该加载哪个能力？",
        "candidates": [
            {"id": "bioinformatics/scanpy", "name": "scanpy", "kind": "skill", "description": "single-cell pipeline QC clustering UMAP"},
            {"id": "bioinformatics/pydeseq2", "name": "pydeseq2", "kind": "skill", "description": "bulk RNA-seq differential expression"},
            {"id": "bioinformatics/anndata", "name": "anndata", "kind": "skill", "description": "annotated matrix data structure"},
            {"id": "bioinformatics/deeptools", "name": "deeptools", "kind": "skill", "description": "NGS signal processing"},
        ],
    })
    print("tool", json.dumps(tool, ensure_ascii=False)[:500])
    kb = call("/v1/decide", {
        "requestId": "smoke-kb", "kind": "kb", "role": "pi",
        "state": "三组处理、每组 4 个样本，要做差异表达；想先确认 FDR 和多重检验的边界。",
        "candidates": [
            {"id": "General Science/multiple-testing-and-false-discovery-rate", "name": "FDR", "kind": "kb", "description": "multiple testing and FDR"},
            {"id": "General Science/experimental-design-and-sample-size", "name": "Design", "kind": "kb", "description": "power and replicates"},
            {"id": "Bioinformatics/pydeseq2", "name": "PyDESeq2", "kind": "kb", "description": "DESeq2 in Python"},
        ],
    })
    print("kb", json.dumps(kb, ensure_ascii=False)[:500])
    role = call("/v1/decide/role_select", {"state": "帮我检索近三年阿尔茨海默病小鼠模型的单细胞转录组文献，整理成证据表。"})
    print("role", json.dumps(role, ensure_ascii=False)[:500])
    relay = call("/v1/decide/relay_select", {
        "from": "researcher", "to": "writer", "why": "结果表和统计已经定稿，写作归 writer。",
        "body": "请基于这些数字写 Results 段，不要改数字，也不要新增结论。",
    })
    print("relay", json.dumps(relay, ensure_ascii=False)[:500])
    ctx = call("/v1/decide/context_retention", {
        "state": "当前任务：修复 DESeq2 报错并重跑差异表达。",
        "groups": [
            {"seq": 12, "kind": "tool-orphan", "tool": "shell", "outcome": "error", "text": "Error in DESeqDataSetFromMatrix: every gene contains at least one zero"},
            {"seq": 13, "kind": "tool", "tool": "file", "outcome": "ok", "text": "counts.tsv 10234 genes x 12 samples"},
            {"seq": 14, "kind": "assistant", "tool": "", "outcome": "ok", "text": "好的，我们开始吧！有问题随时告诉我。"},
        ],
    })
    print("context", json.dumps(ctx, ensure_ascii=False)[:500])
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
