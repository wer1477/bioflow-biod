# issue_owner — 审稿 issue 的责任角色路由（复用 role_select）

## 任务

给定一条审稿 issue，判断它应该由哪个角色处理。输出为 5 类之一。

## 五个角色的职责（判定依据）

| 角色 | 负责 | 不负责 |
|---|---|---|
| `pi` | 研究目标、任务分配、决策与成果验收；跨角色协调、需要仲裁或取舍 | 不替其他角色做具体工作 |
| `searcher` | 文献检索与证据综合、研究现状、假设生成与比较 | 不执行实验、不写正文 |
| `researcher` | 研究方案与实验设计、方法选择、执行、统计、结果解释 | 不做文献综述、不写正文 |
| `writer` | 论文框架与叙事、结果组织、图表、引用、投稿材料 | 不改数据、不跑分析 |
| `reviewer` | 创新性、假设、方法学、数据与统计、结论一致性审查 | 不直接改稿 |

## 路由规则

1. issue 要求**补实验/改统计/改方法** → `researcher`
2. issue 要求**补文献/补证据/改研究现状表述** → `searcher`
3. issue 是**表述、结构、图表、引用格式、夸大措辞** → `writer`
4. issue 是**质疑创新性/假设合理性/结论一致性**，但不需要新实验 → `reviewer`
5. issue 涉及**多角色取舍、资源分配、是否值得改、是否撤稿/改投** → `pi`
6. 真的无法判断 → `pi`（且 confidence ≤ 0.6）

## 输入 `state_text` 写什么

一条完整的审稿 issue：标题 + 前提（作者做了什么/声称什么）+ 证据位置（章节/图表）+ 期望修改。长度 ≤ 1200 字符。

## 输出 schema

```json
{"task":"role_select","id":"io-000001","source":"teacher_synthetic","generator":"issue_owner","difficulty":"medium","lang":"zh","state_text":"...","label":{"role":"writer","confidence":0.85,"reason_code":"owns_manuscript_edit"}}
```

`reason_code` 只能取：

`owns_new_experiment` | `owns_new_search` | `owns_manuscript_edit` | `owns_review_judgement` | `needs_arbitration` | `out_of_scope`

## 生成配额（每个 run）

- 五类角色大致均衡：`pi` 20% / `searcher` 15% / `researcher` 30% / `writer` 20% / `reviewer` 15%
- 难度 easy 30% / medium 45% / hard 25%；语言 zh 60% / en 40%
- 至少 15% 是「跨角色诱惑」样本：表面上像 writer，实际需要 researcher（或反之）

## 示例

```json
{"task":"role_select","id":"io-000001","source":"teacher_synthetic","generator":"issue_owner","difficulty":"easy","lang":"zh","state_text":"标题：多重检验校正缺失。前提：作者称对 12 个终点做了校正，但方法部分未报告校正方法，结果表 p 值均为未校正值，且结论依赖其中 2 个次要终点的显著性。位置：methods 与表 2。期望：补充校正方法并重算，或改写结论强度。","label":{"role":"researcher","confidence":0.88,"reason_code":"owns_new_experiment"}}
{"task":"role_select","id":"io-000002","source":"teacher_synthetic","generator":"issue_owner","difficulty":"medium","lang":"zh","state_text":"标题：图 3 与正文描述不一致。前提：正文称处理组凋亡率显著升高，但图 3 的柱状图误差棒跨越零效应，且图注未说明 n。位置：results 第 2 段与图 3。期望：核对原始数据并统一表述。","label":{"role":"writer","confidence":0.72,"reason_code":"owns_manuscript_edit"}}
{"task":"role_select","id":"io-000003","source":"teacher_synthetic","generator":"issue_owner","difficulty":"hard","lang":"en","state_text":"Title: novelty overlaps with a 2023 paper. Premise: the core method is nearly identical to a prior study, but the application domain differs. Location: introduction and methods. Expected: decide whether the contribution still stands and whether to reposition the paper.","label":{"role":"pi","confidence":0.7,"reason_code":"needs_arbitration"}}
```
