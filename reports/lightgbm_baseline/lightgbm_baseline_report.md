# 第三步：LightGBM 基线

全量 **688,998** 条训练记录、**1,105** 个原始匿名特征，
沿用第二步五折。`label=1` 为还款；所有预测均为还款概率。
整体 OOF 还款 AUC **0.816569**，违约 AP **0.388912**，
相对逻辑回归 AUC 差为 **+0.060609**。
log loss 为 **0.225850**，Brier 为 **0.062500**。
五折 AUC 均值 **0.816578**，标准差（ddof=1）
**0.001843**；这不是官方测试成绩或置信区间。

## 验证协议与早停

每次留出一折作外层最终评价。从其余四折中按标签分层取 **15%**
作为内部早停集，随机种子 `42 + fold`。第一阶段只在内部拟合集建立分箱、训练树，
早停集通过 `reference=inner_data` 使用相同分箱，监控 AUC 选择树数。
预先固定最多 **600** 轮，连续 **50** 轮没有达到
至少 **0.0001** 的有效 AUC 改善则早停。选好树数后，释放第一阶段模型，
在完整拟合四折重新建立分箱、从头训练同样数量的树；不再早停。
最后才预测外层验证折。外层验证数据和标签、测试数据均不参与分箱和树数选择。

达到树数预算上限的折数为 **1/5**。
达到上限只表示本轮计算预算用完，不能宣称已经找到全局最佳树数。
内部早停 AUC 属于用于选择的分数，不替代外层 OOF 指标。

| fold | inner_fit_rows | early_stop_rows | rows | selected_rounds | refit_rounds | selection_hit_round_cap | repay_roc_auc | default_pr_auc_ap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 468518 | 82680 | 137800 | 528 | 528 | False | 0.819397 | 0.390201 |
| 1 | 468518 | 82680 | 137800 | 512 | 512 | False | 0.816091 | 0.389444 |
| 2 | 468518 | 82680 | 137800 | 597 | 597 | True | 0.814260 | 0.382803 |
| 3 | 468519 | 82680 | 137799 | 505 | 505 | False | 0.816463 | 0.395799 |
| 4 | 468519 | 82680 | 137799 | 513 | 513 | False | 0.816679 | 0.387162 |

## 与常数、逻辑回归的相同折外比较

从上一轮保存的 OOF 重新计算参照指标，并核对 id、标签、折号和压力标记完全一致。
违约 AP 定义为 `average_precision_score(1-label, 1-p_repay)`。
ROC-AUC 和 AP 越高越好；log loss 和 Brier 越低越好。

| model | group | rows | default_rate | repay_roc_auc | default_pr_auc_ap | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant | overall | 688998 | 8.3308% | 0.499992 | 0.083307 | 0.286775 | 0.076368 |
| constant | feature_492_upper_tail | 70711 | 9.4950% | 0.498916 | 0.094772 | 0.314695 | 0.086070 |
| constant | feature_1039_observed | 123163 | 9.3494% | 0.501006 | 0.093702 | 0.311204 | 0.084857 |
| logistic_regression | overall | 688998 | 8.3308% | 0.755960 | 0.300177 | 0.247593 | 0.067345 |
| logistic_regression | feature_492_upper_tail | 70711 | 9.4950% | 0.710103 | 0.234618 | 0.289066 | 0.080191 |
| logistic_regression | feature_1039_observed | 123163 | 9.3494% | 0.728794 | 0.259300 | 0.279617 | 0.077564 |
| lightgbm | overall | 688998 | 8.3308% | 0.816569 | 0.388912 | 0.225850 | 0.062500 |
| lightgbm | feature_492_upper_tail | 70711 | 9.4950% | 0.764189 | 0.307811 | 0.269198 | 0.075717 |
| lightgbm | feature_1039_observed | 123163 | 9.3494% | 0.781842 | 0.341513 | 0.259445 | 0.072874 |

两个压力组由第二步提前确定，本次只对子集 OOF 概率计算指标，不重新选阈值或训练。
各组违约率不同，AP 需要结合基准率理解；AUC 也可能随样本组成变化。
随机折不能验证测试集超出训练取值范围的未知标签。

## 固定模型配置

```python
params = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
    "min_data_in_leaf": 200, "lambda_l2": 1.0,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "max_bin": 63, "bin_construct_sample_cnt": 20000,
    "use_missing": True, "zero_as_missing": False,
    "deterministic": True, "force_col_wise": True,
}
```

保留原始 NaN，由每个分裂学习缺失值方向；实际零值不当作缺失。
不做均值填补、标准化、人工缺失指示、异常值剪裁、特征筛选或类别重加权。
测试全空的 7 列仍保留，作为与前一基线一致的初始特征集合。
使用 CPU 4 线程；`max_bin=63` 和 20,000 行分箱样本控制资源开销。
分箱抽样只估计分箱边界；模型阶段使用全部相应拟合行，行采样仅发生在每轮建树。
`IndexedSequence` 分批读取磁盘 float32 原始特征，并按接口要求将当前批转换为 float64。
LightGBM 仍需在内存保存分箱数据和直方图；不是零内存训练。

## 特征重要性

每折完整重训模型的分裂增益先归一化为 gain share，再取五折均值。
split count 为使用该特征分裂的次数。它们描述已训练模型，不是因果效应、系数方向，
也不是排列重要性；相关特征会分摊增益，高基数特征可能更容易被选中。
本轮没有根据该重要性删列或重新调参。

| feature | mean_gain_share | mean_split_count |
| --- | --- | --- |
| feature_96 | 7.24% | 28.4 |
| feature_658 | 6.05% | 152.8 |
| feature_1045 | 5.63% | 52.2 |
| feature_1018 | 2.79% | 24.8 |
| feature_1001 | 2.67% | 13.8 |
| feature_611 | 2.37% | 18.4 |
| feature_447 | 2.19% | 189.2 |
| feature_606 | 2.09% | 39.2 |
| feature_492 | 1.66% | 155.6 |
| feature_421 | 1.54% | 85.4 |
| feature_580 | 1.50% | 148.4 |
| feature_879 | 1.46% | 47.2 |
| feature_192 | 1.33% | 26.2 |
| feature_70 | 0.74% | 25.0 |
| feature_747 | 0.70% | 58.6 |

## 运行产物与复现

- `artifacts/lightgbm_baseline/models/fold_*.txt` 与 JSON：用于 OOF/测试预测的完整四折重训模型及参数。
- `artifacts/lightgbm_baseline/inner_splits/fold_*.npz`：内部拟合、早停、外层验证的原始训练行位置。
- `artifacts/lightgbm_baseline/oof_predictions.parquet`：id、标签、固定折号、压力标记、OOF 还款概率。
- `artifacts/lightgbm_baseline/test_predictions.parquet`：测试 id、五折还款概率均值。
- `submissions/lightgbm_baseline.csv`：与官方模板 id 一对一对齐的本地提交，未上传 Kaggle。
- `reports/lightgbm_baseline/learning_curves/fold_*.json`：内部早停集逐轮 AUC。
- `comparison_metrics.csv`、`fold_metrics.csv`、`feature_importance.csv`、`summary.json`、`run_config.json`：结果与配置。

训练全部阶段耗时 **2590.2 秒**。
记录输入文件来源、折表 SHA256、库版本和训练签名，复用上一轮原始特征缓存。
checkpoint 存储完整折结果，`--resume` 只在配置/来源一致时复用已完成折。
产物核查已通过：内部划分无交叉、外层折表未变、模型重载复现预测、指标与提交对齐均通过。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\scripts\run_lightgbm_baseline.py
.\.venv\Scripts\python.exe .\scripts\verify_lightgbm_baseline.py
.\.venv\Scripts\python.exe .\scripts\plot_lightgbm_baseline.py
```

## 学习用图

下面的曲线只使用内部早停集。圆点标出按 min_delta/patience 规则选出的树数，
不是外层最终评价；达到 600 轮上限的折仍可能受预算限制。

![内部早停曲线](C:/Users/ASUS/Desktop/large-scale-loan-repayment-modeling-challenge/reports/lightgbm_baseline/figures/inner_learning_curves.png)

下面使用全量 OOF 计算 ROC 和 AP，再对曲线渲染顶点抽稀。
PR 图使用阶梯线，避免把常数预测的人工端点连成有预测能力的斜线。

![折外指标对比](C:/Users/ASUS/Desktop/large-scale-loan-repayment-modeling-challenge/reports/lightgbm_baseline/figures/oof_metric_curves.png)


阅读代码与面试准备见 `docs/lightgbm_baseline.md`。
