# SQL 业务分析字段字典

数据库为 `data/interim/sql_business_analysis.duckdb`。全部概率和比例均采用 0–1；报告中的百分比仅改变显示。
训练标签 `1=全额偿还，0=违约`。匿名特征的经济含义未知。

## 表与粒度

| 表/视图 | 一行的含义 | 来源 |
|---|---|---|
| model_registry | 一套固定模型版本 | 现有模型及核查记录 |
| model_predictions | 一个数据集、ID、模型版本的预测 | 冻结 OOF/测试预测 |
| data_quality_summary | 一项已有质量概览指标 | 原库 dq_quality_overview |
| feature_drift_summary | 一个匿名特征的训练—测试漂移 | 原库 dq_train_test_drift |
| model_performance_summary | 一个模型的整体训练 OOF 指标 | SQL 同分聚合与概率误差 |
| risk_band_boundaries | 一个模型的训练 OOF 分位点 | Q10–Q90 |
| risk_band_assignments（视图） | 一条预测加其风险层 | 共用训练边界 |
| risk_band_summary | 数据集 × 模型 × 风险层 | 每组保留全部十层 |
| approval_threshold_grid | 一个固定阈值 | 配置中的预声明网格 |
| approval_policy_results | 数据集 × 模型 × 阈值 | 固定规则的描述性结果 |
| champion_challenger_comparison | 数据集 × 阈值的配对对比 | 同概率阈值 |
| prediction_score_summary | 数据集 × 模型的分数分布 | 预测表 |
| prediction_drift_summary | 一个模型的 OOF—测试分数差异 | 共用风险边界 |

## 身份、标签和模型登记

| 字段 | 定义 |
|---|---|
| dataset | train / test；ID 只在各自数据集内唯一 |
| id | 数据集内关联键，不能作为模型特征或客户时间戳 |
| model_version / model_role | 固定版本；champion 为项目参照，challenger 为候选 |
| prediction_kind | oof / test_ensemble；不使用训练内拟合概率 |
| label | 还款标签；测试为 NULL |
| fold | 固定五折中的 0–4；测试为 NULL |
| stress_feature_492_upper_tail | 既有验证协议中的高尾压力标记；不是风险十分位 |
| stress_feature_1039_observed | 既有非缺失压力标记；测试不提供该训练评价标记 |
| p_repay / p_default | 原始模型还款/违约分数；两者和为 1，未校准 |
| training_signature | 既有训练协议签名 |
| prediction_column | 在原始产物中使用的预测列名 |
| feature_count | 固定模型使用的原始匿名特征数，1,105 |
| oof_models_per_row / test_ensemble_models | OOF 每行/测试预测使用的模型数量（1/5 或 3/15） |
| historical_native_reload_audit | 历史原生模型重载核查状态 |
| current_checkpoint_reconstruction | 当前折预测重建与平均核查状态 |
| max_checkpoint_difference | 当前重建概率与保存概率的最大绝对差 |
| evaluation_scope | reused_development_oof；不是全新独立测试 |
| probability_calibrated / production_deployed | 本阶段均为 false |

## 模型指标、风险层和分数统计

| 字段 | 定义 |
|---|---|
| row_count / labeled_rows | 总行数/有标签行数；测试 labeled_rows 为 0 |
| default_rows / repaid_rows | 标签为 0/1 的人数；测试为 NULL |
| repay_roc_auc | 还款排序 ROC-AUC；同分正负对贡献 0.5 |
| default_pr_auc_ap | 违约 Average Precision，非 PR 曲线梯形积分 |
| ks | 按模型分数排序后两类累计比例的最大绝对差 |
| log_loss / brier_score | 概率误差；log loss 的概率下限为 1e-15 |
| observed_default_rate / observed_repayment_rate | 实际违约/还款人数除以有标签人数；测试为 NULL |
| boundary_reference | train_oof；全部边界由训练 OOF 估计 |
| risk_decile | 1–10，10 为最高违约分数；等于边界归入较低风险层 |
| p10–p90（边界表） | p_default 的训练 Q10–Q90；同分导致分位点可能重复 |
| population_share | 层内总人数 / 同数据集同模型全部人数 |
| mean_model_p_default | 原始违约分数均值，不等于未来实际违约率 |
| min_p_default / max_p_default | 层内实际分数范围；空层为 NULL |
| observed_minus_model_default_rate | 层内实际违约率减原始分数均值，仅有标签时可计算 |
| default_lift / repayment_lift | 层内实际违约/还款率除以同数据集整体对应率 |
| default_capture_share | 层内违约人数 / 全体违约人数 |
| cumulative_default_capture_from_high_risk | 从 D10 累计到当前层的违约人数 / 全体违约人数 |
| mean_p_repay / std_p_repay | 原始还款分数均值/总体标准差 |
| min_p_repay / max_p_repay | 分数最小/最大值 |
| p01 / p10 / p50 / p90 / p99（分布表） | p_repay 的相应分位点；注意与违约分数边界表区分 |
| score_psi | 共用训练风险层占比的 PSI，零占比用 1e-6 下限处理 |
| decile_total_variation | 0.5 × 各层训练—测试占比绝对差之和 |
| interpretation | 分数分布诊断，不能解释成测试标签漂移 |

## 审批阈值与模型对比

| 字段 | 定义 |
|---|---|
| threshold / approval_rule | 固定还款概率阈值；p_repay >= threshold 时获批 |
| policy_status | exploratory_not_selected；本阶段不选择最终策略 |
| total_rows / total_labeled_rows | 全部申请人数/其中有标签人数 |
| approved_rows / rejected_rows | 获批/拒绝的申请人数，包括无标签测试申请 |
| labeled_approved_rows | 获批且有标签的人数；实际获批指标的分母 |
| approval_rate / rejection_rate | 获批/拒绝人数除以 total_rows |
| approved_default_rows / approved_repaid_rows | 获批申请中实际违约/还款人数；测试为 NULL |
| rejected_default_rows | 被拒绝申请的历史违约人数；测试为 NULL，不代表真实减少的违约 |
| observed_default_rate_approved / observed_repayment_rate_approved | 获批实际违约/还款人数除以 labeled_approved_rows；零分母为 NULL |
| population_default_rate | 全体有标签申请的实际违约率 |
| mean_model_p_default_approved | 获批申请的平均原始违约分数，未校准 |
| observed_default_rate_reduction | 全体实际违约率减获批实际违约率，描述人群组成变化 |
| rejected_default_capture_share | 拒绝的历史违约人数 / 全部历史违约人数 |
| comparison_basis | 相同概率阈值比较，不保证相同审批率 |
| champion_version / challenger_version | 对比双方的固定模型版本 |
| champion_* / challenger_* | 双方审批率、实际获批违约率、平均获批原始违约分数 |
| delta_approval_rate / delta_observed_default_rate | Challenger 减 Champion 的比例差，单位为比例/百分点，不是相对增长率 |

质量表沿用原有 PSI、SMD、缺失率差和规则定义，可在 `reports/sql_data_quality_report.md` 查询。
本阶段不建立本金、利息、LGD、模拟净收益或实验分组字段。
