# A/B Test 方案与离线模拟：复现说明

本阶段是求职项目的实验设计交付，不是真实借款人的实验。
用已冻结 Champion/Challenger 审批策略，完成共同入组、随机化契约、ITT 指标、护栏、功效、期限和历史模拟。
结果只适用于固定模型、假设现金流与不变历史标签；不覆盖真实政策改变客户行为、贷款条款或历史获批人群的选择偏差。
OOF 已用于多轮模型开发，当前评价子集不是新的独立模型测试。没有继续训练模型或调整审批阈值。

## 复现

项目根目录 PowerShell：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ab_test_design tests.test_approval_policy_simulation tests.test_sql_business_analysis -v
.\.venv\Scripts\python.exe scripts/run_ab_test_design.py
.\.venv\Scripts\python.exe scripts/verify_ab_test_design.py
```

配置在 `configs/ab_test_design.json`。先只读核查上一阶段，再生成设计与模拟；不连接业务系统。
再次运行会校验并复用同一冻结实验，原分组、模拟随机种子、图表和报告不会重抽。
改变配置或实现协议，使用新名称保留原证据：

```powershell
.\.venv\Scripts\python.exe scripts/run_ab_test_design.py --run-name ab_test_design_v2
.\.venv\Scripts\python.exe scripts/verify_ab_test_design.py --run-name ab_test_design_v2
```

不要并发写同一个 run-name。输入/代码/输出哈希不一致时拒绝复用；完成标记是 `experiment_freeze_manifest.json`。
报告在 `reports/ab_test_design/ab_test_design_report.md`，包括完整实验协议与真实数据演练结果。

## 默认协议与假设

| 项目 | 默认设置 |
|---|---|
| A/B 规则 | 上一阶段冻结模型和完整精度阈值；不重新选择 |
| 真实随机单位建议 | 每借款人首个符合条件申请；稳定 borrower_id，pilot 独立 |
| 离线演示键 | `(dataset,id)`，不能当作已验证借款人标识 |
| 共同代理入组 | `min(raw_score_A,raw_score_B)>=0.90`，不看标签；需要额外真实准入验证 |
| 正式演练范围 | 仅 policy_evaluation 中代理合格训练行；测试只做覆盖诊断 |
| 分流 | SHA256 固定盐值的 Bernoulli A:B=1:1；风险层用于诊断，非精确区组平衡 |
| 主指标 | 每随机入组申请净收益 ITT，包含拒绝成本 |
| 功效 | 双侧 α=0.05、80% 边际功效、示例 MDE=5 元/申请 |
| 方差 | 仅策略开发子集，两策略边际收益方差 ×1.25；不使用配对差值方差 |
| 非劣效护栏 | ITT 违约发生率 B−A<0.2 个百分点；审批率 B−A>−2 个百分点，均看单侧区间 |
| 绝对风险 | B 获批组合违约率 Wilson 单侧上界≤5% |
| 次指标 | 获批还款率、审批率、风险分数层集中度；匿名特征不对应人口属性 |
| 时序 | 固定样本量后一次主分析；提前安全暂停可用，提前收益显著晋级禁止 |
| 周期示例 | 365 天成熟+30 天数据宽限；流量、准入占比和正式实验暴露比例均是假设 |
| Bootstrap/MC | 2,000 次 Bootstrap、1,000 次独立未来样本 Monte Carlo |

MDE 是规划备择效应，不是检验阈值。业务价值点估计筛选另列，不代表置信下界超过业务门槛。
各端点边际 80% 功效不代表全部护栏和主指标同时通过概率为 80%。
全流程“生产晋级就绪”始终 false；shadow 需要确认目标人群分数覆盖、身份、成熟现金流和风险。
原始阈值测试审批率约 17%，训练约 70%；共同代理入组也受漂移影响，不能按历史效果承诺未来收益。

## 统计实现

`experiment_design_common.py` 提供共同代理入组、稳定分流、正态样本量、Welch/非劣效诊断、Wilson 上界、Bootstrap 与 Monte Carlo。

主指标连续收益的独立组方差为 `var_A/n_A+var_B/n_B`。按正态功效搜索最小整数总 N。
非劣效规划使用开发 A 的二项基线与“两个策略发生率相同”的备择假设；相对边界距离为 margin。
零批准风险率保持 NULL；零坏账的 Wilson 上界大于零，不能声称零风险。

两种 Bootstrap 分开命名：

- `paired_historical_policy_difference`：同一历史申请两策略收益差的区间，固定潜在标签假设下的配对问题。
- `independent_arm_offline_replay`：模拟随机分组后分别重采样两个组，描述不同人群的独立组均值差。

基础现金流只有有限离散取值，多项式计数重采样与逐行 IID 有放回 Bootstrap 完全等价。
Monte Carlo 分别从历史两策略经验边际分布抽样，允许模拟大于历史规模的未来样本，但没有新增真实数据。
Monte Carlo 条件于规划比例下的固定 A/B 人数；哈希 Bernoulli 分流的实际组人数会波动，仍须逐组核查成熟有效样本。
零效应校验平移 B 分布使其均值等于 A，保留方差；输出模拟拒绝率及 Monte Carlo 标准误。
无客户/时间信息，IID 和未来分布稳定性无法验证；区间不覆盖模型选择、策略选择和宏观漂移。

## SQL 与文件

数据库：`data/interim/ab_test_design.duckdb`。

| 表 | 粒度与用途 |
|---|---|
| experiment_population | 每数据集/申请一行，两策略分数/决定/假设潜在收益、共同代理资格 |
| eligibility_summary | 开发/评价/测试覆盖、分歧、双批准/双拒绝与历史配对差 |
| experiment_assignments | 每个代理合格策略评价申请的模拟 A/B 分流与实际所选策略的模拟结果 |
| experiment_metrics | 每模拟组 ITT 净收益、审批、违约发生率、获批还款率；真实逾期 NULL |
| risk_score_concentration | 每组/共同分数层的随机人数、批准人数和获批组合占比 |
| assignment_quality | 全体及分层分流人数、预期比例、双侧精确二项 SRM 检查 |
| power_analysis | 端点×MDE/非劣效 margin×分组概率的边际样本需求与实际正态功效 |
| experiment_duration | 假设流量×准入覆盖的入组、成熟和最终分析周期 |
| bootstrap_results | 配对与独立组模拟 Bootstrap 的估计、区间与标准误 |
| monte_carlo_results | 模拟样本规模、经验效应、检验拒绝率、零效应对照与 MC 误差 |

两个行级表导出为 Parquet，其余 8 个汇总表导出 CSV。示例查询在 `sql/05_experiment/02_example_queries.sql`。
`03_production_event_contract.sql` 是未来真实事件表建议，包含 borrower_id、分组概率、规则版本、实际现金流、DPD 与成熟日期；未执行。

`artifacts/ab_test_design/experiment_protocol.json` 保存协议、完整精度策略、统计计划与真实上线阻断条件。
`experiment_freeze_manifest.json` 保存输入、协议、数据库、导出、图表和报告哈希。
两个重复样本 Parquet 用于重现图形和核查。

关键字段：

| 字段 | 解释 |
|---|---|
| common_score / eligible_proxy | 两个原始还款分数的较小值/共同演示资格；未经校准且非安全证明 |
| decision_disagreement | 两套冻结策略是否给出不同审批决定，分组前已可观测 |
| potential_profit_a/b | 不变历史标签与假设条款下的两策略离线净收益；测试 NULL |
| arm / assignment_uniform | 演示分组/确定性哈希产生的均匀数，盐值与实验 ID 需管理 |
| assignment_probability_b | B 分配概率，正式暴露比例另计 |
| assigned_approved | 当前模拟组的冻结策略审批决定 |
| simulated_observed_net_profit | 所选组的历史假设现金流，拒绝仍承担申请成本 |
| simulated_observed_default/repaid | 当前组政策下的历史模拟事件，非真实实验观察 |
| observed_dpd30 / actual_dpd30_rate | 原数据没有对应字段，始终 NULL |
| baseline_default_incidence | 获批违约/全部共同合格开发申请，不能和获批违约率混用 |
| effect_or_margin | 收益规划效应（元）或非劣效边界差（比例） |
| n_a_expected / n_b_expected | Bernoulli 分流下的预期组人数；真实有效样本仍需逐组检查 |
| null_rejection_rate_centered | 平移到零均值差的模拟检验拒绝比例，不是线上错误率实测 |
| simulation_only / production_rollout_ready | 全部产物为模拟；生产就绪 false |

## 可陈述的项目证据

可以说：完成风险政策 A/B Test 方案、共同入组和 ITT 指标设计，独立组功效与样本量评估，SQL 分流质量分析，历史 Bootstrap 和 Monte Carlo 演练，并识别小收益差与长成熟窗口的实验可行性限制。

不能说：已做真实信贷 A/B Test，已证明因果收益提升，已通过生产信用风险审核，或者把历史配对区间当作真实独立随机组区间。
