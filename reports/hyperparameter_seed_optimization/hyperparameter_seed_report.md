# 第三步：超参数与随机种子优化

使用 **688,998** 条训练记录、**36,324** 条测试记录、
**1,105** 个原始 float32 特征。`label=1` 是还款，所有模型输出还款概率。
固定已有五折、原始 NaN 处理和输入精度，比较 4 组联合配置，再对每折选中的配置重训 3 个种子。
共完成 **20 个内部搜索模型、15 个完整四折重训模型**。

三种子集成的开发 OOF AUC 为 **0.820395**，相对原基线
**0.816569** 的变化为 **+0.003825**；
违约 AP 为 **0.395367**，变化 **+0.006454**；
log loss 为 **0.224226**，变化 **-0.001624**。
三种子集成满足预声明的开发推荐规则，可以作为下一步开发候选。
这是重复使用既有折表的开发验证，没有官方测试成绩，也未上传 Kaggle。

## 一、提前固定的实验设计

| name | learning_rate | num_leaves | min_data_in_leaf | lambda_l2 | feature_fraction | bagging_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_extended | 0.050000 | 31 | 200 | 1.000000 | 0.800000 | 0.800000 |
| regularized | 0.050000 | 31 | 400 | 5.000000 | 0.750000 | 0.850000 |
| wider | 0.050000 | 63 | 300 | 5.000000 | 0.800000 | 0.800000 |
| faster_rate | 0.080000 | 31 | 300 | 3.000000 | 0.800000 | 0.800000 |

这是人工制定的有限候选搜索，没有调用 Optuna，也没有搜索整个参数空间。
多项参数同时变化，结果不能证明某个单独参数造成了收益。
原最多 600 轮提高到 **900** 轮；连续 **60** 轮没有
超过 **0.0001** 的有效 AUC 改善时早停。因此本轮也比较了增加训练预算的效果。
`max_bin=63`、分箱抽样 20,000 行、4 线程、原有行列抽样语义均沿用。
20,000 是估计边界的抽样量，不是整体模型只训练 20,000 行；分箱精度本轮没有优化。

每个外层拟合四折内部按标签分层，85% 训练、15% 早停。
四组候选共用内部划分和分箱；早停 Dataset 引用内部训练分箱。
只按内部早停 AUC 选候选，在最佳内部 AUC 的 0.0001 范围内
优先使用候选表中较早的配置。优先级及容忍量都提前固定，不是显著性检验。
选好树数后在完整拟合四折从头重训三个模型，外层验证不传给训练。

## 二、实际选中的配置与树数

| fold | candidate | inner_auc | selected_rounds | evaluated_rounds | hit_round_cap |
| --- | --- | --- | --- | --- | --- |
| 0 | wider | 0.815746 | 595 | 655 | False |
| 1 | regularized | 0.818284 | 794 | 854 | False |
| 2 | wider | 0.818616 | 667 | 727 | False |
| 3 | regularized | 0.817556 | 646 | 706 | False |
| 4 | regularized | 0.815077 | 602 | 662 | False |

内部搜索达到 900 轮上限：**0/20**；选中配置达到上限：
**0/5**（折号：[]）。
达到上限只是用完预算，不代表训练轮数已最优。
不同折独立选择配置，本轮评价的是这个选择流程，并不是全局统一一组参数的单模型。
结果表和代码使用折号 0–4，图中用 1–5，二者是一一对应的同一份固定划分。

![内部搜索与学习曲线](C:/Users/ASUS/Desktop/large-scale-loan-repayment-modeling-challenge/reports/hyperparameter_seed_optimization/figures/inner_search.png)

## 三、完整 OOF 比较

| model | repay_roc_auc | default_pr_auc_ap | log_loss | brier_score |
| --- | --- | --- | --- | --- |
| baseline | 0.816569 | 0.388912 | 0.225850 | 0.062500 |
| tuned_seed_42 | 0.818183 | 0.392057 | 0.225044 | 0.062303 |
| tuned_seed_2026 | 0.818442 | 0.392083 | 0.224956 | 0.062287 |
| tuned_seed_3407 | 0.818287 | 0.392000 | 0.225008 | 0.062300 |
| tuned_seed_ensemble | 0.820395 | 0.395367 | 0.224226 | 0.062114 |

AUC、违约 AP 越高越好，log loss、Brier 越低越好。
违约 AP 使用 `average_precision_score(1-label, 1-p_repay)`；它不是梯形积分的 PR 面积。
OOF 每一行由排除该行所在折的模型预测，三个种子分别完整覆盖全量训练行一次。
总 OOF AUC 不是五折 AUC 的简单平均，因为它还包含不同折样本间的排序比较。

| fold | repay_roc_auc | delta_repay_roc_auc | default_pr_auc_ap | delta_default_pr_auc_ap | log_loss |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.823834 | 0.004437 | 0.397975 | 0.007775 | 0.223193 |
| 1 | 0.819992 | 0.003901 | 0.396028 | 0.006584 | 0.224286 |
| 2 | 0.818841 | 0.004580 | 0.391192 | 0.008390 | 0.224848 |
| 3 | 0.819219 | 0.002756 | 0.400242 | 0.004443 | 0.224052 |
| 4 | 0.820235 | 0.003556 | 0.392120 | 0.004958 | 0.224749 |

三种子集成在 **5/5** 折的 AUC 高于原基线。
这些折的拟合数据相互重叠，不能把折标准差直接当成独立重复实验的置信区间。

## 四、随机种子和概率平均

模型种子提前固定为 **42、2026、3407**，每折实际建树种子是 `seed + fold`。
外层折表不变，内部划分种子固定 `42 + fold`，分箱种子也固定 `42 + fold`。
只改变 `seed`、`bagging_seed`、`feature_fraction_seed`，不改变输入、分箱、超参数和树数。
树数由种子 42 的内部搜索选出并共用，所以不是为每个种子单独寻找最佳树数。

| fold | seed_auc_mean | seed_auc_std_ddof1 | seed_auc_min | seed_auc_max | mean_row_probability_std_ddof0 | mean_pairwise_prediction_correlation |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.821204 | 0.000468 | 0.820666 | 0.821508 | 0.009591 | 0.975010 |
| 1 | 0.818154 | 0.000400 | 0.817705 | 0.818474 | 0.007978 | 0.982431 |
| 2 | 0.816102 | 0.000743 | 0.815467 | 0.816920 | 0.009925 | 0.973535 |
| 3 | 0.817640 | 0.000441 | 0.817180 | 0.818059 | 0.007446 | 0.984415 |
| 4 | 0.818649 | 0.000104 | 0.818536 | 0.818741 | 0.007331 | 0.984810 |

三个完整 OOF AUC 的范围为 **0.818183–0.818442**。
表中的种子标准差描述同一折的建树随机性，不是正式置信区间；三个种子不足以刻画全部随机风险。
不以外层分数挑最高分种子，不搜索混合权重，全部使用等权概率平均。
测试预测是 3 个种子 × 5 折 = **15 个模型平均**，每个权重 1/15。
OOF 每条记录平均三个同折模型的概率，不混入曾经见过该记录的其他折模型。

![种子稳定性与逐折变化](C:/Users/ASUS/Desktop/large-scale-loan-repayment-modeling-challenge/reports/hyperparameter_seed_optimization/figures/seed_stability.png)

![整体指标](C:/Users/ASUS/Desktop/large-scale-loan-repayment-modeling-challenge/reports/hyperparameter_seed_optimization/figures/overall_metrics.png)

## 五、既有压力组

| model | group | rows | default_rate | repay_roc_auc | default_pr_auc_ap | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | feature_492_upper_tail | 70711 | 9.4950% | 0.764189 | 0.307811 | 0.269198 | 0.075717 |
| baseline | feature_1039_observed | 123163 | 9.3494% | 0.781842 | 0.341513 | 0.259445 | 0.072874 |
| tuned_seed_42 | feature_492_upper_tail | 70711 | 9.4950% | 0.766058 | 0.312601 | 0.268321 | 0.075442 |
| tuned_seed_42 | feature_1039_observed | 123163 | 9.3494% | 0.783573 | 0.344657 | 0.258616 | 0.072646 |
| tuned_seed_2026 | feature_492_upper_tail | 70711 | 9.4950% | 0.765863 | 0.310872 | 0.268543 | 0.075534 |
| tuned_seed_2026 | feature_1039_observed | 123163 | 9.3494% | 0.783197 | 0.344176 | 0.258753 | 0.072680 |
| tuned_seed_3407 | feature_492_upper_tail | 70711 | 9.4950% | 0.766071 | 0.312255 | 0.268517 | 0.075532 |
| tuned_seed_3407 | feature_1039_observed | 123163 | 9.3494% | 0.783354 | 0.343785 | 0.258802 | 0.072715 |
| tuned_seed_ensemble | feature_492_upper_tail | 70711 | 9.4950% | 0.768599 | 0.315936 | 0.267485 | 0.075264 |
| tuned_seed_ensemble | feature_1039_observed | 123163 | 9.3494% | 0.785813 | 0.347969 | 0.257808 | 0.072451 |

压力组来自之前的预声明规则，本轮没有根据它们重新选候选。
两组有重叠；不同违约率影响 AP，不能直接把不同组的 AP 当成模型排名。
测试 `feature_492` 大量值超过训练取值范围；随机五折和这些压力组不能证明未见范围的预测有效。

## 六、开发推荐和边界

预声明规则：集成 AUC 比原基线至少高 0.0005，违约 AP 下降不超过 0.001，
log loss 增加不超过 0.0005 才推荐集成。这是工程选择阈值，不是业务批准或统计显著性。
本次推荐为 **tuned_seed_ensemble**。
本轮选择在开发验证上发生，不能把结果称为独立最终测试成绩。
增加种子提升了模型数量与推理成本；生产是否采用需进一步验证收益与成本。

继续使用原始 float32 输入。已有 float32 舍入不会因为批次转换为 float64 而恢复，
`feature_51`、`feature_118` 的原始 DOUBLE 变化是否有用仍未验证。
没有做概率校准、阈值成本选择、时间外推验证，也没有重新删除漂移特征。

## 七、代码、运行与核查

主训练：`scripts/run_hyperparameter_seed_optimization.py`。
核查：`scripts/verify_hyperparameter_seed_optimization.py`。
图表与报告：`scripts/plot_hyperparameter_seed_optimization.py`、
`scripts/report_hyperparameter_seed_optimization.py`。
逐段代码、参数原理和面试回答：`docs/hyperparameter_seed_optimization.md`。

模型在 `artifacts/hyperparameter_seed_optimization/inner_models` 和 `models`；
分割与概率 checkpoint 在 `inner_splits`、`inner_predictions`、`fold_predictions`；
汇总文件为 `oof_predictions.parquet` 和 `test_predictions.parquet`。
三个单种子及集成本地提交为 `submissions/lightgbm_tuned_seed_*.csv`、
`submissions/lightgbm_tuned_ensemble.csv`，按官方模板 id 对齐，未上传。

核查已通过：重载 **20 个内部模型、15 个最终模型**，
重放全部早停选择和候选选择，核对分割无交叉、模型和预测校验值、OOF 恰好覆盖一次、
全部指标与稳定性、三个单种子和集成提交。
内部、OOF、测试重载预测最大差分别为 **0.0**、
**0.0**、**0.0**。
原配置扩展与旧基线在共同训练轮数范围内，内部 AUC 曲线最大差为 **0**；
后续树数变化仍需结合本轮增加的轮数上限和早停耐心理解。
训练阶段记录耗时合计 **17364.2 秒**，
这是各模型耗时之和，含共享 Dataset 的首次构建和预测；并行时不等于墙钟耗时。
已有 LightGBM 基线和特征筛选训练代码没有修改，历史结果可继续核查。
