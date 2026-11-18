# overclaim_flag — 过度声称检测（H7）

## 任务

给定 `claim`（论文主张）与 `evidence`（支撑它的证据/结果），标出 claim 中**超出 evidence 支持范围**的声称类型。可多选，也可为空。

## 五种 flag

| flag | 含义 | 触发例 |
|---|---|---|
| `scope` | 把结论外推到未测试的人群/条件/剂量/组织 | 只测了 A549 细胞，却写「在肺癌中」 |
| `causality` | 用因果语言描述相关性证据 | 只有相关性分析，却写「X 驱动 Y」 |
| `certainty` | 确定性语言超出证据强度 | 小样本单臂研究写「证明/证实」 |
| `generalization` | 从单一数据集/模型泛化到普遍规律 | 单一队列 AUC 写「可作为通用诊断标志物」 |
| `temporal` | 声称长期/持久/始终，但无时间序列证据 | 只随访 6 个月，却写「长期有效」 |

## 判定规则

1. 先看 evidence 实际支持到什么程度（对象、范围、设计强度、时长）。
2. 只标 evidence **明确不支持**的那部分；claim 措辞虽强但与 evidence 匹配 → 不标。
3. 一条 claim 可同时有 1–3 个 flag；完全匹配时 `flags: []`。
4. 不要因为「写作风格激进」就标 flag，必须有**信息层面的越界**。

## 生成配额（每个 run）

- `overclaim` 80%：其中单 flag 50%、双 flag 35%、三 flag 15%
- `clean` 20%：claim 与 evidence 严格匹配，`flags: []`
- 五种 flag 各自出现率 ≥ 12%
- 难度 easy 30% / medium 45% / hard 25%；语言 zh 50% / en 50%

## 输出 schema

```json
{"task":"overclaim_flag","id":"oc-000001","source":"teacher_synthetic","generator":"overclaim","difficulty":"medium","lang":"zh","claim":"...","evidence":"...","context":"...","label":{"flags":["scope","causality"],"confidence":0.86}}
```

硬约束：`claim` ≤ 400 字符、`evidence` ≤ 1200 字符、`context` ≤ 300 字符；全部单行。
`generator` 取值为 `overclaim` 或 `clean`。

## 示例

```json
{"task":"overclaim_flag","id":"oc-000001","source":"teacher_synthetic","generator":"overclaim","difficulty":"easy","lang":"zh","claim":"该抑制剂在所有非小细胞肺癌患者中均可显著延长生存期。","evidence":"在 38 例 EGFR 突变阳性、既往接受过一线治疗的肺腺癌患者中，中位 PFS 为 7.2 个月（单臂 II 期）。","context":"discussion 章节","label":{"flags":["scope","generalization","certainty"],"confidence":0.91}}
{"task":"overclaim_flag","id":"oc-000002","source":"teacher_synthetic","generator":"overclaim","difficulty":"medium","lang":"en","claim":"Higher PD-L1 expression drives resistance to chemotherapy.","evidence":"PD-L1 expression correlated with shorter progression-free survival (Spearman rho=0.31, p=0.02) in a retrospective cohort of 96 patients.","context":"results section","label":{"flags":["causality"],"confidence":0.88}}
{"task":"overclaim_flag","id":"oc-000003","source":"teacher_synthetic","generator":"clean","difficulty":"easy","lang":"zh","claim":"在本次纳入的 96 例回顾性队列中，PD-L1 高表达与更短的无进展生存期相关（rho=0.31）。","evidence":"96 例回顾性队列，PD-L1 高表达组中位 PFS 5.1 个月 vs 低表达组 8.4 个月，Spearman rho=0.31，p=0.02。","context":"results 章节","label":{"flags":[],"confidence":0.9}}
```
