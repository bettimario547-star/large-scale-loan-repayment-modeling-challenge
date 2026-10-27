# 项目交付导航 / Delivery index

[中文 README](../README.md) · [English README](../README.en.md) · [复现入口](reproducibility.md)

本项目已完成本地模型、SQL 业务分析、审批模拟、严格 A/A、V2 A/B 可行性及决策展示。交付结论为**保留 Champion，暂不晋级 Challenger**。历史 OOF、模拟假设和真实业务待验证项分别记录；没有线上随机实验或真实收益提升。

## 按阅读目的进入

| 读者 / 目的 | 建议顺序 |
| --- | --- |
| 招聘方快速了解 | [README](../README.md) → [一页决策备忘录](experiment_closeout_decision_memo.md) → [Power BI](../dashboard/powerbi/LoanRisk.pbip) |
| 机器学习审阅 | [模型卡](model_card.md) → [验证策略](validation_strategy.md) → [LightGBM](lightgbm_baseline.md) → [有限调参及种子集成](hyperparameter_seed_optimization.md) |
| SQL / 业务分析审阅 | [业务字段字典](sql_business_data_dictionary.md) → [SQL 分析](sql_business_analysis.md) → [审批模拟](approval_policy_simulation.md) |
| 实验设计审阅 | [实验卡](experiment_card.md) → [严格 A/A](aa_validation.md) → [V2 可行性](ab_feasibility_v2.md) → [结项决定](experiment_closeout_decision_memo.md) |
| 工程复现 | [架构](architecture.md) → [复现指南](reproducibility.md) → [单一 CLI](../scripts/project.py) → [交付核查](delivery_verification.json) |
| English review | [README](../README.en.md) → [Model card](model_card.en.md) → [Experiment card](experiment_card.en.md) |

## 实现与证据对应

| 阶段 | 实现 / 配置 | 结果 / 核查 | 说明 |
| --- | --- | --- | --- |
| 数据工程 | [DuckDB 入口](../scripts/init_duckdb.py)、[Parquet](../scripts/convert_to_parquet.py)、[质量](../scripts/run_data_quality.py)、[SQL](../sql/01_ingestion) | [质量报告](../reports/data_quality_report.md) | [数据访问](data_access.md) |
| 验证与 EDA | [EDA](../scripts/run_eda.py)、[行序检查](../scripts/check_train_row_order.py)、[对抗验证](../scripts/run_adversarial_validation.py)、[折号](../scripts/build_validation_strategy.py) | [验证策略报告](../reports/validation_strategy/validation_strategy_report.md) | [验证策略](validation_strategy.md)、[ID 规则](id_feature_policy.md) |
| 模型 | [线性基线](../scripts/run_linear_baselines.py)、[Champion](../scripts/run_lightgbm_baseline.py)、[Challenger](../scripts/run_hyperparameter_seed_optimization.py) | [基线记录](../reports/lightgbm_baseline/run_config.json)、[集成记录](../reports/hyperparameter_seed_optimization/run_config.json) | [模型卡](model_card.md)、[English](model_card.en.md) |
| SQL 风险分析 | [入口](../scripts/run_sql_business_analysis.py)、[SQL](../sql/03_analysis)、[配置](../configs/sql_business_analysis.json) | [汇总](../reports/sql_business_analysis/summary.json)、[核查](../reports/sql_business_analysis/artifact_verification.json) | [SQL 说明](sql_business_analysis.md)、[字段字典](sql_business_data_dictionary.md) |
| 审批模拟 | [入口](../scripts/run_approval_policy_simulation.py)、[SQL](../sql/04_policy_simulation)、[配置](../configs/approval_policy_simulation.json) | [汇总](../reports/approval_policy_simulation/summary.json)、[核查](../reports/approval_policy_simulation/artifact_verification.json) | [模拟说明](approval_policy_simulation.md) |
| 原 A/B 方案 | [入口](../scripts/run_ab_test_design.py)、[SQL](../sql/05_experiment)、[配置](../configs/ab_test_design.json) | [汇总](../reports/ab_test_design/summary.json)、[核查](../reports/ab_test_design/artifact_verification.json) | [原方案](ab_test_design.md)，保留作为版本变更依据 |
| 严格 A/A | [入口](../scripts/run_aa_validation.py)、[配置](../configs/aa_validation.json) | [汇总](../reports/aa_validation/summary.json)、[核查](../reports/aa_validation/artifact_verification.json) | [校验范围与未认证压力检查](aa_validation.md) |
| V2 可行性与再模拟 | [入口](../scripts/run_ab_feasibility.py)、[SQL](../sql/06_ab_feasibility)、[配置](../configs/ab_feasibility_v2.json) | [汇总](../reports/ab_feasibility_v2/summary.json)、[核查](../reports/ab_feasibility_v2/artifact_verification.json)、[预分析锁](../artifacts/ab_feasibility_v2/preanalysis_design_lock.json) | [完整方案](ab_feasibility_v2.md)、[实验卡](experiment_card.md)、[English](experiment_card.en.md) |
| 决策备忘录 | [一页 Markdown](experiment_closeout_decision_memo.md)、[PDF](../output/pdf/experiment_closeout_decision_memo.pdf) | [来源及视觉核查](../reports/experiment_closeout/closeout_provenance.json) | 已冻结，文档补全不重写其历史结果 |
| Power BI | [PBIP](../dashboard/powerbi/LoanRisk.pbip)、[聚合 CSV](../dashboard/powerbi/data)、[生成器](../scripts/build_powerbi_dashboard.py) | [数据来源](../dashboard/powerbi/source_manifest.json)、[静态核查](../dashboard/powerbi/artifact_verification.json) | [使用说明](powerbi_dashboard.md)，主页面已由用户确认刷新成功 |
| 总体交付 | [事实与源哈希](project_facts.json)、[环境](environment_snapshot.json)、[锁定包快照](../requirements-lock.txt) | [文档核查](delivery_verification.json)、[入口测试](../tests/test_project_delivery.py) | [只读业务核查](../scripts/replay_frozen_business.py) |

实际 Python 实现在 `scripts/`，SQL 在 `sql/`，业务配置在 `configs/`。仪表盘使用 Power BI；本交付没有模型 API 或 Streamlit 网站。

## 保留的建模探索说明

[线性基线](modeling_baselines.md) · [特征筛选](feature_screening.md) · [LightGBM](lightgbm_baseline.md) · [超参数与种子](hyperparameter_seed_optimization.md) · [XGBoost/CatBoost](xgboost_catboost_comparison.md) · [竞赛 AUC 探索](public_auc_optimization.md)

这些材料说明开发过程；当前业务模型角色以模型卡和冻结配置为准。复现入口只重建冻结业务所依赖的基线与集成，不自动运行其余探索。

## 交付范围与缺口

| 状态 | 内容 |
| --- | --- |
| 已完成 | 可读代码/SQL、冻结本地预测和分析、离线校准与模拟、卡片、架构、复现 CLI、六页 PBIP、结项材料 |
| 已验证的 Desktop 范围 | 用户确认主页面打开、刷新及度量显示；其余五页尚未逐页确认，PBIX 保存未确认 |
| 真实实施待完成 | borrower_id/首申请、可信时间与安全准入、实际执行/现金流/成熟 DPD、shadow、独立 pilot、锁定主实验、生产服务及监测 |

当前本地具备 Parquet 与冻结业务产物，缺原始 train/test CSV。代码分享时，大型数据和模型默认被忽略；小型 Power BI 聚合 CSV 允许随项目保留。只含源码的副本可审阅文档和聚合结果，但不能声称已重算私有数据/模型；[复现指南](reproducibility.md)区分两种环境。尚未初始化 Git、发布仓库或发布数据。
