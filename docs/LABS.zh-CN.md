# 从零理解 Chat to Dashboard

这是一套可以逐个模块动手完成的实验。目标是让你能解释一次查询为什么正确、失败在哪里、如何比较改进，而不只是运行一个演示。完成代码运行不等于掌握；每关都需要留下你自己的解释和改动记录。

先按 README 安装，在项目目录运行命令。以下数据和结果属于本次教学复现，不代表历史实习项目。

## 0 建立整体地图

```bash
dashboard-lab ask "Revenue by month for 2025"
```

先只看输出里的 `question`、`sql`、`rows`、`chart` 和 `trace`。画出七个节点，标出每个节点读取、修改的状态。再打开 `engine.py`，找到 `StateGraph`、条件边、终点和重试边。

**交作业：**不看代码，用两分钟解释用户提问后到图表出现之间发生什么。必须说明路由、检索、SQL 生成、SQL 执行的区别。

**面试追问：**LangGraph 在这里解决了什么？如果只有一个 prompt 能生成 SQL，为什么还拆节点？拆分带来的延迟、状态复杂度和维护成本是什么？这个版本是节点工作流，哪些步骤真正调用了 LLM？

## 1 先成为这套数据库的分析师

读 `data.py` 和 `assets/catalog.json`，画出六张表的主外键关系，标注一对多关系。订单有多个商品行，也可能有多条退款记录。

手算：某订单有两行商品，折后金额分别 100 和 50；两条退款分别 10 和 5。正确净收入为 135。如果直接把两张明细表按订单关联，会形成四行，收入和退款都会重复。先按订单分别聚合，再关联，才能保留各自粒度。

**交作业：**写一条故意错误的净收入 SQL、一条正确 SQL，比较结果；解释 `COUNT(*)` 与 `COUNT(DISTINCT order_id)` 的不同。说明退款是按订单月还是退款发生月归属。

**验收：**你能解释为什么 `SUM(DISTINCT amount)` 不能普遍修复重复关联：两笔真实交易可能金额相同。

## 2 业务口径与本体

收入、毛销售额、净收入、客单价、退款率和 MRR 的分母、时间、状态要求分别是什么？把每个指标映射到表、字段、关联和约束。

```bash
dashboard-lab ask "Revenue by month"
dashboard-lab ask "Net revenue by month"
dashboard-lab ask "Net revenue by category"
```

第三条需要澄清，因为退款只有订单级别，尚未定义到品类的分摊规则。

**交作业：**新增一个“退款发生月现金流”指标。先写业务定义、金标准 SQL 和边界测试，再加入模板。不能直接复用净收入的时间口径。

## 3 路由与澄清

```bash
dashboard-lab ask "Show customer retention"
dashboard-lab ask "Delete all orders"
dashboard-lab ask "2025年每月收入"
```

分别解释为什么返回 `clarify`、`rejected`、`ok`。再问一个带复杂日期或筛选条件的问题，检查系统有没有默默忽略条件。

**交作业：**添加五个系统目前不能正确处理的自然语言问题，把期望动作写清楚。先让测试失败，再修改路由；不要先看现有模板抄出“容易过”的测试。

**面试追问：**路由标签是业务意图还是结果状态？macro F1 和 micro F1 分别如何计算？本仓库评估的是响应状态，不能直接等同于原简历的 intent-routing F1。

## 4 动态示例检索

读 `retrieval.py`，跟踪一句问题如何变成向量、相似度如何排序、前几个示例如何进入上下文。本地默认用词法哈希向量，理解它与预训练语义 embedding 的差异：词面相近不保证业务含义相同，词面不同也可能是同一指标。

**交作业：**为“客户实际贡献”“退款后收入”等表达检查检索结果，记录命中的示例、表和指标。比较同义词、中文、缩写以及不相关关键词的情况。

**验收：**能说明 224 个模板变体为什么不是 224 个独立任务，为什么精确字符串不重叠仍可能存在模板或语义泄漏。

## 5 业务图与多跳关联

```bash
dashboard-lab ask "Net revenue by region for 2025" --retrieval graph
dashboard-lab ask "Net revenue by region for 2025" --retrieval vector
```

为这个问题画出路径：净收入 → 订单明细与退款；地区 → 客户；各表通过订单与客户标识连接。观察图模式是否补全必要表，也有没有多检索不需要的表。

**交作业：**列出两种模式的 precision、recall 和一个错误案例；删除一条业务关系后重跑，解释变化。

**面试追问：**图扩展增益来自图边、别名识别、更多上下文还是全部因素？严谨消融实验应保持模型、问题、prompt 预算和别名规则一致。本地模板 SQL 不依赖检索结果，所以不能拿它的 SQL 正确率证明 GraphRAG 带来回答质量提升。

## 6 真正的 LLM 生成与修复

离线模板先提供可重复的基线。配置模型后才运行 `--mode llm`。读 `providers.py`，检查发送给模型的业务上下文与返回 JSON 的契约。

**交作业：**在自己的环境配置模型，先用五个简单问题，再用五个含糊问题。保存 SQL、是否澄清、耗时、错误和调用成本。密钥保留在环境变量，不提交仓库。

故意通过测试替身返回一个不存在的字段，观察第一次执行失败如何进入修复，第二次失败为什么必须终止。修复预算是按整次运行计算，不能在不同节点反复重置。

**面试追问：**SQL 能执行就代表业务正确吗？安全检查能否保证答案正确？检索内容中的恶意指令怎样处理？模型超时和 schema 改名如何区分？

## 7 SQL 安全与可观察性

读 `sql_safety.py`。分别解释 AST 检查、只读数据库、SQLite authorizer、表与函数白名单、返回行数限制、时间与指令预算各解决什么问题。

**交作业：**写出三个无破坏的失败实验：查系统表、两个 SQL 语句、代价很高的连接。确认被哪个层拦截。将错误归入“意图、检索、生成、权限、执行、结果、展示”。

**面试追问：**仅在 prompt 中写“不要删除数据”够不够？加 `LIMIT 100` 就能限制聚合计算成本吗？记录完整问题和查询结果可能带来哪些数据处理要求？本项目本地服务不保存问题日志。

## 8 金标准与回归评测

```bash
dashboard-lab evaluate --output artifacts/evaluation.json
python scripts/check_evaluation.py artifacts/evaluation.json
python -m unittest discover -s tests -v
```

逐个看案例的 `gold_sql`、`sql`、实际行、期望行和状态。结果比较要处理行顺序、浮点容差、NULL 和重复行；字段别名不同通常不影响业务结果。

**交作业：**独立写十个从未用于改代码的新问题，冻结后再评测；明确测试时间和是否看过结果。刻意加入“可以执行但答案错”的 SQL。

**指标：**执行成功率 = 成功执行的可回答案例 / 全部可回答案例；结果准确率 = 结果符合金标准的可回答案例 / 全部可回答案例。被错误澄清的问题也留在分母。历史“88% answer-quality parity”还需另行定义评审标准，不能直接复用本仓库结果准确率。

## 9 图表推荐与画布

打开网页，分别查看月份折线、品类柱状图和一个汇总值。展开表格，对收入分组点击 Inspect，确认明细的维度和年份与上层一致。下钻最多显示 100 条，不能把截断后的金额和当成完整汇总。

读 `chart_model.py`，解释训练特征、标签生成、训练与测试划分、最近质心预测和规则回退。600 个合成特征样本只是教学训练过程。测试中有重复特征模式，准确率不能代替真实用户的零修改接受率。

**交作业：**收集至少 20 个不同真实分析任务的匿名表结构，人工选择图表并记录理由；记录“推荐正确”和“用户无需修改”是否一致。设计时间到洞察的起止点与同任务对照，避免把速度改进完全归因于模型。

## 10 工程流程与讲解

GitHub Actions 会在 push 和 PR 运行测试与回归集，并保留评测报告。CI 通过说明当前测试集通过，不能代表所有用户问题都正确。

**交作业：**新建分支完成一个真实功能，例如新增季度分组。先写金标准、修改实现、更新样例、运行回归，最后准备 PR，解释对旧行为的影响。不要只提交截图。

**最终演示顺序：**用一个跨表业务问题开场 → 说明指标口径 → 展示检索上下文 → 执行 SQL → 展示图表与明细 → 故意触发一个澄清 → 展示一个失败案例 → 解释评测分母和目前限制。

英文骨架：

> I built this public learning reconstruction using synthetic commerce data. A LangGraph workflow separates routing, context retrieval, SQL generation, validation, execution, and visualization. The default generator uses deterministic templates, and a configurable LLM adapter handles generation and one bounded repair attempt. The main correctness challenge is preserving metric definitions and join grain. I evaluate executable SQL separately from result correctness, and I treat the included fixtures as development regressions rather than evidence of production accuracy.

这段只描述本次真实完成的复现。讲历史实习时，还需要补上你当时实际负责的部分、团队协作、数据来源和可核实的指标口径。
