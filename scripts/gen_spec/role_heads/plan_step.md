# plan_step — 计划步骤的工具选择（复用 tool_select）

## 任务

给定一个**实验设计/执行计划步骤**，从 researcher 可见工具集中选出一个最该加载的工具；没有适用工具则选 `NO_TOOL`。

## 输入 `state_text` 写什么

一个自洽的步骤描述，包含：
1. 这一步的目标（要产出什么）
2. 已有输入（数据/文件/上一步产物）
3. 约束（算力、是否可联网、是否需要可复现、数据敏感等级）
4. 期望产出形式

长度 ≤ 780 字符（渲染时会被截断到 800，超了会丢信息）。

## 标签 `label.id`

- 必须是工具清单里的 id，或 `"NO_TOOL"`。
- 工具清单见同目录 `_catalog_researcher.txt`（100 个，格式 `id | name | description`）。
- **只写 id，不要写名字**。

## 生成配额（每个 run）

| 子类型 | 占比 | 说明 |
|---|---|---|
| `direct` | 40% | 步骤明确对应一个工具，约束宽松 |
| `constraint` | 30% | 有约束（本地 CPU、不可联网、需要容器化复现、数据敏感），正确答案是更受约束友好的工具 |
| `no_tool` | 20% | 纯思考/写作/沟通/结果解读，不需要任何工具 → `NO_TOOL` |
| `ambiguous` | 10% | 两个工具都沾边，正确答案是更专门的那个 |

难度 easy 30% / medium 45% / hard 25%；语言 zh 60% / en 40%（Agent 真实输入以中文为主）。

## 输出 schema

```json
{"task":"tool_select","id":"ps-000001","source":"teacher_synthetic","generator":"plan_step","difficulty":"medium","lang":"zh","role":"researcher","state_text":"...","candidateIds":["@preset:researcher"],"label":{"id":"bioinformatics/pydeseq2"}}
```

- `candidateIds` **固定写** `["@preset:researcher"]`，由构建脚本展开为完整可见集，不要自己列。
- `label.id` 必须在 researcher 可见集内，否则整条作废。

## 示例

```json
{"task":"tool_select","id":"ps-000001","source":"teacher_synthetic","generator":"plan_step","difficulty":"easy","lang":"zh","role":"researcher","state_text":"目标：对 12 例 bulk RNA-seq 样本做差异表达分析，已有 counts 矩阵与样本分组表，计算资源为本地 32 核 CPU、无 GPU，结果需可复现并输出火山图与富集表。","candidateIds":["@preset:researcher"],"label":{"id":"bioinformatics/pydeseq2"}}
{"task":"tool_select","id":"ps-000002","source":"teacher_synthetic","generator":"plan_step","difficulty":"medium","lang":"zh","role":"researcher","state_text":"目标：确认目前对单细胞多组学整合方法的最新评测结论，形成一个 5 条要点的综述段落。已有：研究问题与关键词。约束：不跑分析、只需要文献，不能联外网。","candidateIds":["@preset:researcher"],"label":{"id":"NO_TOOL"}}
{"task":"tool_select","id":"ps-000003","source":"teacher_synthetic","generator":"plan_step","difficulty":"hard","lang":"en","role":"researcher","state_text":"Goal: estimate tumor purity and ploidy from whole-exome CNV segments for 40 FFPE samples. Inputs: segmented copy-number calls and allele fractions. Constraint: on-prem only, no cloud upload, must produce per-sample purity estimates with confidence intervals.","candidateIds":["@preset:researcher"],"label":{"id":"bioinformatics/cancer-genomics-analysis"}}
```
