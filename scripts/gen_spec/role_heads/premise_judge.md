# premise_judge — 配对关系判定（H6）

## 任务

给定 `left`（待检验命题）与 `right`（证据/原文），判定 `right` 对 `left` 的关系。

## 五种标签

| 标签 | 定义 |
|---|---|
| `supports` | 同一对象、同一范围、同一测量上直接支持；不需要额外假设 |
| `partially_supports` | 方向一致，但只覆盖一部分（人群/条件/指标/时间），或需额外假设才成立 |
| `contradicts` | 针对**同一关系**给出相反方向的结果 |
| `unrelated` | 同领域但不构成证据（不同构念、不同人群、背景综述、纯方法描述） |
| `insufficient` | 谈的正是 left，但无法判定（样本不足、未报效应量、结论含糊、只有相关性无方向） |

## 判定顺序（必须按顺序执行）

1. `right` 在谈 `left` 的同一个关系吗？否 → `unrelated`
2. 是 → 方向相反吗？是 → `contradicts`
3. 方向一致 → 覆盖完整吗？完整 → `supports`；部分 → `partially_supports`
4. 方向无法判断（无数据/无结论）→ `insufficient`

## 边界规则

- **空结果**：样本充足且检验了同一关系 → `contradicts`；样本不足或未检验 → `insufficient`
- **体外 vs 体内**、不同细胞系/物种：构念相同但范围变窄 → `partially_supports`
- **综述/背景段落**提到 left 但没有定量结果 → `unrelated`
- **相关性 vs 因果**：left 是因果主张而 right 只有相关证据 → `partially_supports`
- **同一结论在不同亚组**：亚组一致 → `supports`；仅部分亚组 → `partially_supports`
- 语言不同（中/英）不影响判定

## 置信度

`0.90–0.99` 明确；`0.70–0.89` 少量歧义；`0.55–0.69` 边界样本；低于 0.55 不要生成。

## 三种 unit

| unit | left 是什么 | context 写什么 |
|---|---|---|
| `hypothesis` | 一条可检验的研究假设（来自 gap） | 研究空白 + 领域 + 预期方向 |
| `claim` | 论文里的一句主张 | 所在章节（abstract/results/discussion…）+ claim 类型 |
| `issue` | 审稿人 issue 的前提（如「作者称已做多重校正但未报告方法」） | 章节 + 稿件类型 |

## 生成配额（每个 run）

- `hypothesis_evidence` 40% / `claim_evidence` 35% / `issue_passage` 25%
- 五类标签大致均衡：`supports` ≥25%、`unrelated` ≥20%、`insufficient` ≥20%、`contradicts` ≥15%、`partially_supports` ≥15%
- 难度 easy 30% / medium 45% / hard 25%；语言 zh 50% / en 50%

## 输出 schema

```json
{"task":"premise_judge","id":"pj-000001","source":"teacher_synthetic","generator":"hypothesis_evidence","difficulty":"easy","lang":"zh","unit":"hypothesis","left":"...","right":"...","context":"...","label":{"relation":"supports","confidence":0.95}}
```

硬约束：`left` ≤ 700 字符、`right` ≤ 1400 字符、`context` ≤ 400 字符，全部单行。

## 示例

```json
{"task":"premise_judge","id":"pj-000001","source":"teacher_synthetic","generator":"hypothesis_evidence","difficulty":"easy","lang":"zh","unit":"hypothesis","left":"在 EGFR 19 外显子缺失的肺腺癌中，三代 EGFR-TKI 一线治疗的 PFS 优于含铂化疗。","right":"某 III 期随机对照试验纳入 556 例 EGFR 19del 肺腺癌患者，一线随机接受三代 TKI 或含铂化疗；TKI 组中位 PFS 18.9 个月，化疗组 10.2 个月（HR 0.46，95%CI 0.35-0.61）。","context":"研究空白：三代 TKI 在 19del 亚组的一线获益是否独立于共突变状态。","label":{"relation":"supports","confidence":0.95}}
{"task":"premise_judge","id":"pj-000002","source":"teacher_synthetic","generator":"claim_evidence","difficulty":"medium","lang":"en","unit":"claim","left":"Knockdown of gene X reduced tumor volume by 60% in the xenograft model.","right":"In the same xenograft model, gene X knockdown showed no significant difference in tumor volume versus control (n=8 per arm, p=0.62, 21 days).","context":"results section; claim type: causal","label":{"relation":"contradicts","confidence":0.93}}
{"task":"premise_judge","id":"pj-000003","source":"teacher_synthetic","generator":"issue_passage","difficulty":"hard","lang":"zh","unit":"issue","left":"作者声称对 12 个终点做了多重检验校正，但方法部分未报告校正方法或校正后的 p 值。","right":"方法：主要终点为 OS，次要终点包括 PFS、ORR、DCR 等共 12 项；统计部分仅写明采用双侧检验，显著性水平 0.05。结果表 2 报告的 p 值均为未校正值。","context":"methods 章节；稿件为单中心回顾性研究","label":{"relation":"supports","confidence":0.88}}
{"task":"premise_judge","id":"pj-000004","source":"teacher_synthetic","generator":"claim_evidence","difficulty":"medium","lang":"zh","unit":"claim","left":"该模型在外部数据集中同样达到 AUC 0.85，说明其泛化能力良好。","right":"外部验证集仅 42 例、来自同一家医院、同一测序平台，AUC 0.83（95%CI 0.68-0.94）。","context":"discussion 章节；claim 类型：generalization","label":{"relation":"partially_supports","confidence":0.79}}
```
