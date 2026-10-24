# 缺失模式、漂移和低方差特征筛选

使用完整 688,998 条训练记录、原始 1,105 特征，沿用固定分层五折。
`label=1` 是还款，所有模型预测还款概率，违约 AP 使用 `1-label` 与 `1-p_repay`。

## 实验设计

本轮四个方案在训练前声明，单独与原始 LightGBM 比较，不把多个变更堆叠后归因。
各折固定上一轮内部早停选定并完整重训的树数：[528, 512, 597, 505, 513]。
固定原有模型参数和随机种子，不在本轮重新选树数或调参。当前折的规则仅由其余四折拟合；
外层验证不用于规则学习、分箱、叶子拟合或树数选择。测试集的预先声明列名单来自以前的
无标签诊断，是利用测试协变量的针对性消融；没有用测试值学习填补、阈值或还款模型。

- `missing_patterns`：保留原始值和 NaN；在拟合四折内计算还款/违约缺失率差，绝对差至少
  0.10 的列增加二值缺失标记，另加原始 1,105 列的逐行缺失数量。
- `availability_filter`：删除预先确定的测试全空 7 列，再删除拟合四折缺失率至少 90% 的列。
  这是可用性组合方案，不能单独归因于其中某一种删除。
- `drift_core_filter`：删除预先声明的 `feature_492/903/586/1039`。前三列为此前主要来源
  分类与数值漂移候选，1039 为缺失率剧变候选。不是删除所有 541 个高漂移标记列。
- `low_information_filter`：检查精确众数/缺失状态占比，至少 99.5% 的拟合行处于同一状态
  才删除。NaN 单独算一个状态。非缺失值恒定但缺失模式变化的列仍可能保留。
  原始非缺失方差只报告，不按统一阈值删除匿名且尺度不同的特征。无列可删时复用原模型。

## 整体 OOF

| variant | repay_roc_auc | auc_delta | default_pr_auc_ap | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.816569 | 0.000000 | 0.388912 | 0.225850 | 0.062500 |
| missing_patterns | 0.816381 | -0.000189 | 0.388462 | 0.225892 | 0.062506 |
| availability_filter | 0.816343 | -0.000227 | 0.388955 | 0.225878 | 0.062490 |
| drift_core_filter | 0.815981 | -0.000588 | 0.387376 | 0.226145 | 0.062586 |
| low_information_filter | 0.816407 | -0.000163 | 0.388595 | 0.225885 | 0.062504 |

本轮按整体 OOF AUC 排名的本地开发最优是 **baseline**，AUC **0.816569**。
这是当前验证数据上的方案比较与选择，不能把选中后的分数当作新的完全独立测试结果。
五折差异只作为配对描述，不当作五个独立实验做显著性保证。

## 压力组

| variant | group | rows | default_rate | repay_roc_auc | default_pr_auc_ap |
| --- | --- | --- | --- | --- | --- |
| baseline | overall | 688998 | 0.083308 | 0.816569 | 0.388912 |
| baseline | feature_492_upper_tail | 70711 | 0.094950 | 0.764189 | 0.307811 |
| baseline | feature_1039_observed | 123163 | 0.093494 | 0.781842 | 0.341513 |
| missing_patterns | overall | 688998 | 0.083308 | 0.816381 | 0.388462 |
| missing_patterns | feature_492_upper_tail | 70711 | 0.094950 | 0.763965 | 0.308869 |
| missing_patterns | feature_1039_observed | 123163 | 0.093494 | 0.781309 | 0.341484 |
| availability_filter | overall | 688998 | 0.083308 | 0.816343 | 0.388955 |
| availability_filter | feature_492_upper_tail | 70711 | 0.094950 | 0.764378 | 0.309278 |
| availability_filter | feature_1039_observed | 123163 | 0.093494 | 0.781579 | 0.340744 |
| drift_core_filter | overall | 688998 | 0.083308 | 0.815981 | 0.387376 |
| drift_core_filter | feature_492_upper_tail | 70711 | 0.094950 | 0.762218 | 0.301627 |
| drift_core_filter | feature_1039_observed | 123163 | 0.093494 | 0.779616 | 0.335902 |
| low_information_filter | overall | 688998 | 0.083308 | 0.816407 | 0.388595 |
| low_information_filter | feature_492_upper_tail | 70711 | 0.094950 | 0.764013 | 0.308552 |
| low_information_filter | feature_1039_observed | 123163 | 0.093494 | 0.781735 | 0.341298 |

压力组沿用既有标记，用同一批 OOF 子集计算，不单独训练或选阈值；各组可重叠、违约率不同。
随机划分仍不能验证测试中超出训练范围且没有标签的取值。删除漂移列也不保证消除整体漂移。

## 每折特征规则

| variant | fold | removed | indicators | row_count_added | output_features | identity |
| --- | --- | --- | --- | --- | --- | --- |
| missing_patterns | 0 | 0 | 114 | True | 1220 | False |
| missing_patterns | 1 | 0 | 108 | True | 1214 | False |
| missing_patterns | 2 | 0 | 118 | True | 1224 | False |
| missing_patterns | 3 | 0 | 107 | True | 1213 | False |
| missing_patterns | 4 | 0 | 114 | True | 1220 | False |
| availability_filter | 0 | 8 | 0 | False | 1097 | False |
| availability_filter | 1 | 8 | 0 | False | 1097 | False |
| availability_filter | 2 | 8 | 0 | False | 1097 | False |
| availability_filter | 3 | 8 | 0 | False | 1097 | False |
| availability_filter | 4 | 8 | 0 | False | 1097 | False |
| drift_core_filter | 0 | 4 | 0 | False | 1101 | False |
| drift_core_filter | 1 | 4 | 0 | False | 1101 | False |
| drift_core_filter | 2 | 4 | 0 | False | 1101 | False |
| drift_core_filter | 3 | 4 | 0 | False | 1101 | False |
| drift_core_filter | 4 | 4 | 0 | False | 1101 | False |
| low_information_filter | 0 | 2 | 0 | False | 1103 | False |
| low_information_filter | 1 | 2 | 0 | False | 1103 | False |
| low_information_filter | 2 | 2 | 0 | False | 1103 | False |
| low_information_filter | 3 | 2 | 0 | False | 1103 | False |
| low_information_filter | 4 | 2 | 0 | False | 1103 | False |

每折规则可以不同，预测必须使用该折保存的 feature plan 和列顺序。
`FeatureView` 在请求的行/批次上应用规则，原始 NaN/零值保留，不复制完整转换矩阵。
LightGBM 的分箱 Dataset 和直方图仍在内存；特征采样、行采样与数据分批是不同操作。

## 产物与核查

配置和训练签名在 `run_config.json`；完整折内统计在 `fold_feature_statistics.csv`；
精确删列原因和新增列在 `feature_plans.json`；各折指标在 `fold_metrics.csv`，配对差在
`paired_fold_deltas.csv`。每方案保存模型/规则、OOF、五折测试概率均值与本地提交，均未上传。
缺失特征的重要性单独记录，但不是因果效应。无变化方案明确记录复用而非伪造新训练。
训练耗时 4220.1 秒。产物核查通过：模型重载复现全部预测、重新拟合规则一致、OOF 每行覆盖一次、指标及提交对齐。

![特征处理消融对比](C:/Users/ASUS/Desktop/large-scale-loan-repayment-modeling-challenge/reports/feature_screening/figures/ablation_comparison.png)


学习说明见 `docs/feature_screening.md`。这些实验回答本地验证下的收益和代价，未建立时间验证，
也没有证明修复了训练/测试漂移。

## 输入精度核查

| feature | original_unique | float32_unique | original_dominant_state | float32_dominant_state |
| --- | --- | --- | --- | --- |
| feature_118 | 3049 | 51 | 2.7909% | 99.6331% |
| feature_51 | 494 | 25 | 97.4794% | 99.5157% |


全训练原始 DOUBLE 与基线 float32 缓存的对照说明，相近值的舍入可以明显增加状态集中度。
这一核查用于解释筛选候选，没有改动折内名单。本轮各方案均采用基线相同输入精度，
不能把 float32 中的近常数结论直接归给原始高精度列。是否恢复高精度需要单独的输入实验。

![折内特征诊断](C:/Users/ASUS/Desktop/large-scale-loan-repayment-modeling-challenge/reports/feature_screening/figures/fold_feature_diagnostics.png)

## 新增缺失列的训练重要性

| fold | derived_gain_share | row_missing_count_gain_share |
| --- | --- | --- |
| 0 | 0.5716% | 0.0265% |
| 1 | 0.8158% | 0.0253% |
| 2 | 0.5064% | 0.0147% |
| 3 | 1.5342% | 0.0133% |
| 4 | 1.0216% | 0.0135% |

gain 占比只描述训练分裂对损失的改善，不表示新增特征的折外收益或因果效应。
