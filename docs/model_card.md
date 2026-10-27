# 模型卡：贷款全额偿还预测

[English](model_card.en.md) · [项目入口](../README.md) · [实验卡](experiment_card.md)

版本日期：2026-10-25。用途为 Kaggle 概率预测、历史风险分析和离线策略比较；**当前业务决定保留 Champion，模型未进入生产审批服务**。本卡针对业务冻结的两套 LightGBM，不把其它 Kaggle 探索候选自动视为业务 Challenger。

## 模型登记

| 项目 | Champion | Challenger |
| --- | --- | --- |
| model_version | `lightgbm_baseline` | `lightgbm_tuned_seed_ensemble` |
| 算法 | 二分类 LightGBM 基线 | 四组预声明联合配置的内部搜索 + 三种子均值 |
| 外层折数 | 5 | 5 |
| 模型随机种子 | 42（各折加 fold） | 42 / 2026 / 3407（各折加 fold） |
| 最终四折重训模型 | 5 | 15 |
| 训练 OOF | 每条申请的外层折模型预测 | 同一外层折三种子预测均值 |
| 无标签测试 | 五折模型均值 | 五折 × 三种子的 15 模型均值 |
| 冻结还款阈值 | 0.9247938423619432 | 0.9256811726550405 |
| 当前角色 | 保留的业务参照 | 开发候选，未晋级 |

登记与 SHA-256 见[业务配置](../configs/sql_business_analysis.json)、[冻结模型清单](../artifacts/sql_business_analysis/model_freeze_manifest.json)、[模型登记表](../reports/sql_business_analysis/tables/model_registry.csv)。Challenger 训练还包含 20 个内部候选搜索模型；业务冻结采用最终预测所需的 20 个重训模型，不包括这 20 个搜索模型。

## 输入、目标与预处理

688,998 个训练申请，36,324 个无标签测试申请，1,105 个匿名数值特征。目标 `label=1` 表示全额偿还，`0` 表示违约。输出为 `p_repay`；违约分数为 `1-p_repay`。

- 输入集合来自严格 `feature_*` 白名单；`id`、标签、fold 和压力标记排除。ID 仅用于对齐和提交。
- LightGBM 使用原始 float32 特征、原生 NaN 缺失分支；真实零不视为缺失，无均值填补、标准化或人工缺失指示。
- 原始概率未校准。平均违约分数不等于已验证的真实未来违约概率。
- 数据缺少可信贷款日期、期限/观察窗口、持久借款人标识和可解释人口属性。不能解释匿名特征为收入、年龄或地域。

## 训练与验证设计

固定 `StratifiedKFold(5, shuffle=True, random_state=42)`，折表保存为 `validation_folds.parquet`。训练文件行序未建立可信时间意义，因此没有把末段行号称为未来留出。

每折从拟合四折内分层留出 15% 作早停；先用内部拟合建立分箱/训练，内部早停集选树数，再用全部拟合四折重训；外层第五折不参与该轮早停。Challenger 在相同内部划分与分箱上比较四组配置，按内部 AUC 和预声明同分优先级选配置/树数；所有三个重训种子共享树数，未按结果挑幸运种子。该搜索不是 Optuna 贝叶斯优化。

基线最多 600 轮、耐心 50；Challenger 最多 900 轮、耐心 60。基线 `learning_rate=0.05`、`num_leaves=31`、`min_data_in_leaf=200`、L2=1、行/列采样 0.8、`max_bin=63`。逐折选择和树数见[基线记录](../reports/lightgbm_baseline/run_config.json)、[调参记录](../reports/hyperparameter_seed_optimization/run_config.json)及[训练说明](hyperparameter_seed_optimization.md)。

**开发 OOF 已被多轮模型/策略迭代复用。** 内部早停保持本轮外层隔离，不会让整个开发历史成为新独立确认性评估。业务层再拆开发/评价子集也不能恢复此前消耗的独立性。

## 历史表现

| 指标 | Champion | Challenger |
| --- | ---: | ---: |
| 还款 ROC-AUC | 0.816569 | 0.820395 |
| 违约 PR-AUC（AP） | 0.388912 | 0.395367 |
| KS | 0.472156 | 0.479461 |
| Log loss | 0.225850 | 0.224226 |
| Brier score | 0.062500 | 0.062114 |

AUC 绝对差 +0.003825。这是 688,998 行复用 OOF 的开发比较；没有测试标签，不能给测试集计算上述真实性能，也没有据此推导真实净收益或因果提升。指标与 SQL 同分实现、sklearn 复算见[结果汇总](../reports/sql_business_analysis/summary.json)及[独立核查](../reports/sql_business_analysis/artifact_verification.json)。

训练/测试分布漂移很强；例如行序/验证诊断记录了大量测试 `feature_492` 非缺失值超过训练取值上界。冻结阈值在无标签测试上的审批率约 17%，与评价数据约 70% 差异明显。这些是分布/分数诊断，测试实际坏账率仍未知。压力分组不能验证训练中未出现且没有标签的测试区域。

## 策略与使用限制

冻结策略由 413,398 个开发申请按基准经济收益和约束选定；275,600 个评价申请用于描述，均使用历史 OOF。两套策略的开发审批预算均为 70%，规则为 `p_repay >= frozen_threshold`。准入、风险和实验仍需[实验卡](experiment_card.md)的独立约束。

不把排序增益当作晋级条件。V2 离线模拟收益差 +3.672 元、95% 区间 [0.119, 7.226]；业务要求下界 >5 元，未达到。模型/策略不直接授权真实审批、拒绝或放量。

已知限制包括重复客户相关性不可核查、时间外推不充分、分数未校准、真实现金流与期限不可用、审批选择/标签覆盖机制未知，以及匿名数据不支持有意义的公平性分组评估。重要性是模型诊断，不是因果或业务字段解释。

## 复现与运行约束

```powershell
python scripts/project.py reproduce --target models --dry-run
.\.venv\Scripts\python.exe scripts/project.py verify --scope business
```

模型阶段需先完成数据/折号及线性基线，去掉 `--dry-run` 才实际训练。业务核查复用冻结预测，不重新读取全部特征推理。模型重载推理的历史证据另保存在模型阶段核查文件，不能将当前哈希核查称为新重载推理。

模型阶段的历史环境为 LightGBM 4.7.0 / NumPy 2.5.3；当前审阅环境 NumPy 2.5.4。固定种子、输入和代码签名帮助重现，不保证跨环境模型字节一致。已有签名不匹配时保留旧产物，在干净工作副本建立新实验。完整步骤见[复现指南](reproducibility.md)。

生产前还需可信时间队列、borrower_id 去重、概率校准和校准验证、成本/现金流、漂移与局部安全监测、模型服务/版本日志和成熟实验结果。本项目没有交付上述生产链路。
