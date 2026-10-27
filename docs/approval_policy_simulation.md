# 审批策略模拟：运行、假设与字段说明

本阶段基于已经冻结的两套 LightGBM 模型，实现审批策略的历史离线经济模拟。
配置和假设是模拟参数，不是 Kaggle 数据真实合同字段，也不是实际经营结果。

## 运行

项目根目录 PowerShell：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_approval_policy_simulation tests.test_sql_business_analysis -v
.\.venv\Scripts\python.exe scripts/run_approval_policy_simulation.py
.\.venv\Scripts\python.exe scripts/verify_approval_policy_simulation.py
```

运行先只读核查上一步 SQL 业务层，再读取已冻结预测。没有训练模型、重新推理或概率校准。
重复运行复用冻结的开发/评价划分与策略阈值；输入、经济配置或策略协议变更时用新名称：

```powershell
.\.venv\Scripts\python.exe scripts/run_approval_policy_simulation.py --run-name approval_policy_simulation_v2
.\.venv\Scripts\python.exe scripts/verify_approval_policy_simulation.py --run-name approval_policy_simulation_v2
```

参数文件：`configs/approval_policy_simulation.json`。修改假设后使用新 run-name，保留原有结果。
不要并发写同一个 run-name。旧模型、冻结输入和 SQL 分析库不被改写。

## 经济模型

基础假设：本金 10,000 元、12 个月利息收益率 12%、LGD 80%、同期限资金成本率 4%、
每笔获批服务成本 100 元、每个申请成本 10 元。金额假设为人民币，数据实际币种未知。

本金偿还不是收益。获批且还款取得全部设定利息；获批且违约损失 LGD 比例本金，不计利息。
资金与服务成本由全部获批贷款承担，申请成本由全部申请承担，包括拒绝申请。
基准单笔净收益：还款 690 元、违约 -8,510 元、拒绝 -10 元。

```text
net_i = approve_i × [label_i × principal × interest_rate
                    − (1−label_i) × principal × LGD
                    − principal × funding_rate − servicing_cost]
        − application_cost
```

同期限利息与资金成本率直接对应 12 个月，未模拟摊还或不同违约时点。
`historical_profit_per_application` 的分母是全体申请，便于下一阶段规划 ITT 实验指标。
未经校准分数代入期望式所得收益独立命名为 `raw_score_implied_*`，不作为本次选择依据。

## 策略开发与选择

固定模型后，将已有训练 OOF 按标签分层划为 60% 策略开发、40% 策略评价，种子 20261025。
两模型共享 ID 划分。底层 OOF 曾参与模型开发，所以这一划分只隔离本次策略选择，不能构成新独立模型测试。

在开发分数上拟合相同目标审批预算：0%、30%、40%、50%、60%、70%、75%、80%、85%、90%、95%、100%。
除全批/全拒端点外，按还款分数从高到低设阈值，规则为 `p_repay >= threshold`。
同分边界会超预算时整组排除，不用 ID 任意拆同分客户。实际批准人数不超过 floor(开发人数 × 目标审批率)。
评价与测试使用开发得到的固定阈值，不在各自数据上重新拟合配额，因此审批率可以变化。

基础情景中，开发获批违约率 ≤ 5%、开发审批率 ≥ 50% 的候选才有资格。
每模型选择每个申请历史模拟净收益最高的候选；利润四舍五入至六位小数，平手依次按
较低违约率、较低审批率、policy_id 排序。若无可行候选，脚本失败，不强行生成策略。
这只是开发点估计约束，不是风险置信上界或生产安全保证。

选择后保存 A（Champion）和 B（Challenger），以及全批/全拒控制参照。
6 个经济情景评价相同冻结策略，不在各情景重选。相同预算对比表明确记录实际批准人数是否相等。

## 数据库与字段

数据库：`data/interim/approval_policy_simulation.duckdb`。7 份表导出在 `reports/approval_policy_simulation/tables/`。

| 表/视图 | 粒度与用途 |
|---|---|
| economic_assumptions | 一个经济情景；base 用于选择，其他情景用于单参数敏感性 |
| policy_candidates | 一个模型 × 开发目标审批预算；含阈值、规则与开发批准人数 |
| policy_constraints | 一行开发风险上限与审批率下限 |
| policy_cohorts | 每个训练 ID 的共享开发/评价归属 |
| simulation_predictions（视图） | 冻结模型预测加 policy_cohort；测试统一标记 unlabeled_test |
| approval_economic_results | 情景 × 候选策略 × 数据群组的风险和现金流 |
| matched_approval_comparison | 相同目标审批预算的 A/B 模型对比；不强制评价阶段人数相同 |
| selected_policy_definitions | A、B、全批、全拒的冻结定义；prospective_arm 不代表已随机分组 |
| selected_policy_results | 选定策略在各情景、群组上的结果 |
| selected_policy_application_outcomes（视图/Parquet） | 基础情景下每策略、数据集、ID 的批准与模拟净收益，用于下一阶段 |

| 字段 | 定义 |
|---|---|
| policy_cohort | policy_development、policy_evaluation 或 unlabeled_test |
| scenario_id / currency / horizon_months | 模拟情景、假设币种、公共观察期限 |
| principal / interest_rate_for_horizon / loss_given_default | 假设统一本金、同期限利息率、违约本金损失比例 |
| funding_rate_for_horizon / servicing_cost_per_approved / application_cost_per_application | 获批资金成本率、获批服务成本、全部申请成本 |
| incremental_break_even_p_repay | 与拒绝相比的审批盈亏平衡概率；申请成本因两种选择均支付而相消 |
| policy_id / model_version / model_role | 固定策略、来源模型和参照/候选身份 |
| target_approval_rate / threshold / decision_rule | 开发预算、开发阈值、score_threshold/approve_all/reject_all |
| development_approved_rows / development_total_rows | 阈值拟合时实际批准人数/开发总人数 |
| total_applications / labeled_applications | 全体申请/其中有标签人数 |
| approved_applications / rejected_applications | 批准/拒绝人数，包含无标签测试申请 |
| approved_repaid_rows / approved_default_rows | 获批历史还款/违约人数；测试为 NULL，训练零获批为 0 |
| approval_rate / observed_default_rate_approved | 批准人数/总人数；实际违约人数/有标签获批人数 |
| approved_score_sum / mean_p_repay_approved | 获批还款分数合计/均值，未校准 |
| historical_simulated_interest_income / historical_simulated_principal_loss | 用历史标签计算的模拟利息/本金损失；测试为 NULL |
| assumed_funding_cost / assumed_servicing_cost / assumed_application_cost | 按批准或全部申请人数计算的假设成本 |
| historical_simulated_net_profit | 历史标签模拟利息减本金损失与全部成本；测试为 NULL |
| historical_profit_per_application | 历史模拟净收益/全部申请；包含拒绝成本 |
| historical_profit_per_approved_loan | 组合净收益/批准人数，包含全部申请成本；零批准为 NULL |
| raw_score_implied_net_profit / raw_score_implied_profit_per_application | 分数代入期望式的诊断值；不是校准后的预期利润或未来实收 |
| eligible_for_selection | 仅 base/开发计算的风险和审批约束通过状态，其他群组为 NULL |
| max_development_default_rate / min_development_approval_rate | 预声明开发选择约束 |
| prospective_arm | A、B 或控制参照名；不是 experiment_assignment |
| approved | 逐申请固定策略决定 |
| exactly_equal_approved_count | 相同目标预算下，两模型在当前群组批准人数是否完全相等 |
| champion_* / challenger_* / delta_* | 双方批准人数、审批率、获批实际违约率和每申请净收益；delta 为 B 减 A |

无标签测试的所有历史利润和实际坏账指标为 NULL；全拒的获批坏账率为 NULL，不写成零风险。

## 报告、核查与交接

报告：`reports/approval_policy_simulation/approval_policy_simulation_report.md`，包括审批率—违约率、
收益—风险、经济敏感性、ROC/PR 与 Lift/Gains 四张图。

核查：SQL 与逐申请 NumPy 现金流一致；政策选择只用开发；测试无标签指标为 NULL；
实际等量状态、敏感性、选定策略逐申请收益、CSV/Parquet 和 SHA256 均核验。

冻结文件位于 `artifacts/approval_policy_simulation/`：开发/评价 ID、全部候选阈值、A/B/控制策略，
`policy_freeze_manifest.json` 记录来源、协议与文件哈希。
`experiment_policy_handoff.json` 交接已选规则、评价指标、逐申请收益路径与待设计事项；当前没有随机分流。

本次运行中，固定阈值的评价审批率约 70%，测试审批率仅约 17%；交接文件同时保留两个群组的审批率，
标记 `shadow_validation_required=true` 与 `production_rollout_ready=false`。
后续实验设计须先安排 shadow mode 检查目标人群分数分布、审批量与成熟标签风险。
测试集没有真实结果，不能用审批量变化推断违约率，也不能为满足配额直接在测试分数上重拟合阈值。

核查时，金额的 SQL/NumPy 浮点求和允许相应数值容差；仅原始分数隐含收益总额的绝对容差为 0.001 元，
相对容差为 1e-11。人数严格一致；逐申请现金流的绝对容差为 1e-8 元。
CSV 按稳定业务键排序核对，保留空值布尔类型和浮点数值，Parquet 则逐字段精确核对。

必须保留假设边界：历史还款标签在审批策略与贷款条款改变后保持不变；原数据可能仅覆盖原本获批贷款，
并不必然代表全部真实申请。匿名特征、未知合同条款、训练—测试漂移和无结果时间都限制未来解释。
模拟净收益不是实际业务收入，政策对比不是因果估计，未来 A/B Test 仍需设计入组、安全、随机化、功效和成熟窗口。
