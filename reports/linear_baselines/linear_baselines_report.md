# 第三步：常数预测与逻辑回归基线

## 本次结果

使用全量 **688,998** 条训练记录、全部 **1,105** 个匿名特征，
沿用第二步保存的五折；每条记录只有一次折外预测（OOF）。`1=还款，0=违约`。
逻辑回归整体折外还款 ROC-AUC 为 **0.755960**，
违约 PR-AUC（Average Precision）为 **0.300177**。
常数基线的每折 ROC-AUC 都是 0.5；逻辑回归提供了可比较的线性模型起点。
本次五折优化器全部报告收敛：**True**。
逻辑回归五折 AUC 均值为 **0.755964**，
标准差（ddof=1）为 **0.001609**。

## 整体和压力分组的折外指标

| model | group | rows | default_rate | repay_roc_auc | default_pr_auc_ap | log_loss | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant | overall | 688998 | 8.3308% | 0.499992 | 0.083307 | 0.286775 | 0.076368 |
| constant | feature_492_upper_tail | 70711 | 9.4950% | 0.498916 | 0.094772 | 0.314695 | 0.086070 |
| constant | feature_1039_observed | 123163 | 9.3494% | 0.501006 | 0.093702 | 0.311204 | 0.084857 |
| logistic_regression | overall | 688998 | 8.3308% | 0.755960 | 0.300177 | 0.247593 | 0.067345 |
| logistic_regression | feature_492_upper_tail | 70711 | 9.4950% | 0.710103 | 0.234618 | 0.289066 | 0.080191 |
| logistic_regression | feature_1039_observed | 123163 | 9.3494% | 0.728794 | 0.259300 | 0.279617 | 0.077564 |

`overall` 是整体；`feature_492_upper_tail` 是第二步预先保存的高尾组；
`feature_1039_observed` 是其非缺失组。分组没有重新拟合模型或重新选阈值。
PR-AUC 在这里明确定义为 `average_precision_score(1-label, 1-p_repay)`，
并非 PR 曲线的梯形积分。分组违约率不同，AP 也受基准发生率影响，不能只比较 AP。
ROC-AUC、AP 越高越好；log loss、Brier 越低越好。

逻辑回归在高尾组的 AUC 为 **0.710103**，
在非缺失组为 **0.728794**，均低于整体。
两组仍优于各自的常数参照，但存在分组性能下降，后续模型应继续使用相同压力组比较。

## 每折结果与优化状态

| model | fold | repay_roc_auc | default_pr_auc_ap | log_loss | converged | iterations |
| --- | --- | --- | --- | --- | --- | --- |
| constant | 0 | 0.500000 | 0.083309 | 0.286778 | True | 0 |
| logistic_regression | 0 | 0.757623 | 0.303616 | 0.246887 | True | 227 |
| constant | 1 | 0.500000 | 0.083309 | 0.286778 | True | 0 |
| logistic_regression | 1 | 0.756615 | 0.299338 | 0.247681 | True | 284 |
| constant | 2 | 0.500000 | 0.083309 | 0.286778 | True | 0 |
| logistic_regression | 2 | 0.753449 | 0.298038 | 0.248459 | True | 282 |
| constant | 3 | 0.500000 | 0.083302 | 0.286762 | True | 0 |
| logistic_regression | 3 | 0.755419 | 0.303786 | 0.247292 | True | 294 |
| constant | 4 | 0.500000 | 0.083310 | 0.286779 | True | 0 |
| logistic_regression | 4 | 0.756713 | 0.297206 | 0.247646 | True | 279 |

五折 ROC-AUC 均值与标准差、优化器迭代次数、终止原因和梯度大小保存在
`summary.json`、`fold_metrics.csv` 和各折模型 JSON。均值/标准差是五折描述统计，
不把它当作独立样本的置信区间。

## 常数基线：先确认评价指标没有误导我们

每折仅用拟合四折的 `label.mean()`，为验证折所有行输出同一个还款概率。
五折常数略有不同，所以拼接后的整体 OOF AUC 可能略低于 0.5，
这是折间基准概率差造成的排序现象，不表示常数有预测能力。
全量训练还款率是 **91.669207%**。
常数模型按 0.5 阈值全部预测还款，准确率仍有 **91.669207%**；
这个准确率不能证明模型识别了违约。测试常数提交使用全量训练还款率。

## 逻辑回归：缺失填充、标准化和 L2 正则

每折用拟合四折的非缺失均值填补缺失，再使用填补后拟合数据的标准差缩放。
填补后的缺失值在标准化空间为 0。填补参数和缩放参数都不使用验证折或测试集。
本基线不加入缺失指示变量、不按标签筛特征、不剪裁异常值、不使用类别权重。
保留全部 1,105 列（包括测试全空的 7 列），作为明确的初始参照；
测试全空列只能被均值填充，后续可单独做缺失指示与删列消融。
`id`、`label`、`fold` 和压力标记均不进入 X。

模型输出 `sigmoid(X @ coef + intercept)`，含义是还款概率。
固定 `C=1.0`，最小化：

```text
mean(logaddexp(0, score) - label * score) + ||coef||² / (2 * C * n_fit)
```

截距不惩罚，目标与 scikit-learn 的二分类 L2 `LogisticRegression` 一致。
使用 SciPy L-BFGS-B 做全批量优化，分块累加损失和梯度；不是抽样训练或 SGD。
特征落盘为 float32，统计量、参数、损失累加为 float64；矩阵乘法存在 float32
舍入误差。小型测试与 sklearn 预测进行数值对照。
终止上限为 500 次，梯度容差为 0.0001；
这些参数在评价前固定，不使用外层验证折早停或调参。

## 文件与复现

- `artifacts/linear_baselines/oof_predictions.parquet`：id、标签、固定折号、两模型折外概率和压力标记。
- `artifacts/linear_baselines/test_predictions.parquet`：测试 id、常数还款概率、五折逻辑回归概率均值。
- `artifacts/linear_baselines/models/fold_*.npz`：各折系数、截距、填补均值、缩放参数；JSON 记录特征顺序和优化状态。
- `submissions/constant_baseline.csv`、`submissions/logistic_regression_baseline.csv`：按官方模板 id 一对一对齐的本地提交文件。
- `reports/linear_baselines/summary.json`、`summary_metrics.csv`、`fold_metrics.csv`、`run_config.json`：完整结果和运行配置。
- `reports/linear_baselines/artifact_verification.json`：另运行产物核查脚本后生成的检查结果。

逻辑回归测试预测来自五折模型的均值，没有另做全量重训。文件已生成，未上传 Kaggle，
因此没有官方测试集分数。缓存使用约 5.7 GB 磁盘，避免把宽表整体放入内存。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\scripts\run_linear_baselines.py
.\.venv\Scripts\python.exe .\scripts\verify_linear_baselines.py
```

本次运行耗时 **739.5 秒**。训练与折表来源、折表 SHA256、
库版本和完整配置保存在 `run_config.json`。再次运行会重新训练，复用来源匹配的特征缓存。
产物核查已通过：折表未变、OOF 对齐正确、指标重算一致、五折模型重载后复现测试概率、两份 CSV 与模板 id 和保存概率一致。


## 结论的边界

这是训练分布内的线性基线。第二步已发现训练/测试强漂移；尤其大量测试
`feature_492` 值超出训练取值范围，压力组依旧不能验证那部分未知标签。
训练内较高的 AUC 不能等同于 Kaggle 表现。
下一阶段可在相同折号上训练树模型，再独立检验缺失指示和漂移列消融的作用。
