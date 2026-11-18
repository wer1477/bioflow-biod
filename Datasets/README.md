# Datasets — Paper 1 canonical store

本目录是 paper 1（9 决策 / 7 头）在 deployment-host 上的**唯一数据入口**。
数据 payload 不纳入 git（`.gitignore` 的 `Datasets/v*/`）；旧路径 `artifacts/datasets/`
已归档到 `other/archive-pre-paper1-20261103/datasets/artifacts-datasets/`。

## 1. 9 决策 / 7 头训练集

| # | 决策点 | 头 | 训练文件 | train / val / test |
|---|---|---|---|---|
| 1 | `tool_select` | tool | `v3/mixed/tool_select_*.jsonl`（v2 598 + plan_step 320） | 918 / 116 / 116 |
| 2 | `kb_select` | kb | `v2/multitask/kb_select_*.jsonl` | 720 / 90 / 90 |
| 3 | `context_retention` | context | `v2/multitask/context_retention_*.jsonl` | 442 / 54 / 54 |
| 4 | `role_select` | role | `v3/mixed/role_select_*.jsonl`（v2 720 + issue_owner 261） | 981 / 122 / 122 |
| 5 | `relay_select` | relay | `v2/multitask/relay_select_*.jsonl` | 640 / 80 / 80 |
| 6 | `r1.plan-audit` | tool（复用） | 同上 `v3/mixed/tool_select_*` | 918 / 116 / 116 |
| 7 | `s2.premise` | premise | `v3/multitask/premise_judge_*.jsonl` | 850 / 105 / 105 |
| 8 | `w1.claim-evidence-audit` | flag + premise | `v3/multitask/overclaim_flag_*.jsonl` + `premise_judge_*` | 493 / 61 / 61（flag） |
| 9 | `v1.issue-triage` | premise + role（复用） | 同上 `v3/mixed/role_select_*` + `premise_judge_*` | 981 / 122 / 122 |

7 个训练头：`tool / kb / context / role / relay / premise / flag`。
`r1` 复用 tool，`v1` 的 owner 复用 role，`w1/v1/s2` 共用 premise；不设第 8 个 `reason` 头。

## 2. 验收与一致性

| 集合 | 规模 | 位置 |
|---|---|---|
| premise_judge gold | 200 | `v3/role_heads/gold/premise_judge.jsonl` |
| overclaim_flag gold | 200 | `v3/role_heads/gold/overclaim_flag.jsonl` |
| 双标一致性探针 | 40 / 60 | `v3/role_heads/kappa/` |
| `by-decision/` | 9 个决策点的软链 | 指向 v2/v3 文件 |

`v3/role_heads/raw/`、`merged/` 保留，作为 `build_role_heads.py` 的输入与数据 provenance；
`v0/`、`v1/`（4B 时代）已归档，不再使用。

## 3. 冻结口径

- catalog：`v2/catalog_snapshot.json`，`catalogHash = 3fdeb703…`。
- 切分种子：`20261102`；test 只写一次；`manifest.json` 的 `rows + sha256 + catalogHash` 是权威。
- 2B/4B/9B 三个容量共用本目录的同一份训练集；不得按容量换数据。

## 4. 新实验数据

新的配对扰动集将写入 `Datasets/paper1/`（待建）：base scenario + scope/constraint 变体 +
长度控制 + item covariates + 人审子集。不要修改 `v2/`、`v3/` 原文件。
