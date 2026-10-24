# 第三步建模路线：LightGBM 基线学习说明

实现见 [`scripts/run_lightgbm_baseline.py`](../scripts/run_lightgbm_baseline.py)，
实际结果见 [`reports/lightgbm_baseline/lightgbm_baseline_report.md`](../reports/lightgbm_baseline/lightgbm_baseline_report.md)。
本项目 `label=1` 为还款，`label=0` 为违约。模型输出 `p_repay`，不是违约概率。

## 1. 这一轮要回答什么问题？

上一轮逻辑回归的整体 OOF AUC 为 0.755960、违约 AP 为 0.300177。
逻辑回归对原始输入学习线性加权分数，本轮换成梯度提升树，检查通过阈值分裂、
非线性关系和特征交互能否改善预测。所有模型沿用同一份分层五折与两个压力组，
比较时不把一次不同划分造成的波动当作模型进步。

此前对抗验证也用过 LightGBM，但它的目标是“这行来自训练集还是测试集”。
本轮的目标是“这笔贷款是否还款”。那次来源分类的 AUC 不能当作还款预测成绩，
也不能直接使用那次的模型生成还款提交。

本轮使用全量 688,998 行训练数据、全部 1,105 个匿名原始特征。
`id`、`label`、`fold` 和压力标记不进入 X；没有根据当前结果筛列或搜索参数。

## 2. 从一棵树理解模型

一棵决策树通过条件把样本分到不同叶子。例如在一个假想业务场景里：

```text
收入 < 某个阈值？
  是：再判断负债水平
  否：再判断历史还款情况
```

这只是说明原理。本项目特征匿名，不能把某个 `feature_*` 宣称为收入或负债。
同一特征可以多次按阈值分裂；不同特征沿同一路径组合，形成特征交互。
叶子存储的是对当前模型分数的更新值，不应把 Boosting 的每个叶子直接理解成
该叶所有样本的还款率。

LightGBM 默认采用 leaf-wise 生长：每次优先分裂能带来较大目标改善的叶子，
因此树可能不平衡。`num_leaves` 限制叶子数，`min_data_in_leaf` 限制小叶子，
降低追着少量样本拟合的风险。

本次 fold 0 完整重训模型的第一棵树可以提供一个真实例子：根节点判断
`feature_644 <= 3.4070276`，根节点缺失值走右侧；左子节点判断
`feature_96 <= 0.2930733`，该节点缺失值走左侧；右子节点判断
`feature_760 <= 5.0005286`，该节点缺失值也走左侧。这仅展示树的前两层，
后面仍会继续分裂。这棵树共有 31 个叶子。匿名编号不能解释为某个确定的业务字段。
运行绘图脚本会从实际模型导出 `reports/lightgbm_baseline/tree_example.json`。

## 3. 多棵树怎样合作？

模型先从总体还款比例对应的初始分数出发，再依次训练树来改善已有模型。
二分类时的形式是：

```text
score(x) = initial_score + learning_rate * (tree_1(x) + ... + tree_T(x))
p_repay(x) = sigmoid(score(x))
```

新树的学习方向来自当前损失对分数的一阶梯度、二阶梯度（Hessian）。
二分类 log loss 的一阶梯度是 `p_repay - label`；不是简单把错分的行重复训练，
也不是把每棵树的概率平均。本轮五折模型最终对测试概率取平均，是另一层模型组合。

`objective='binary'` 决定训练用二分类 log loss；`metric='auc'` 决定内部早停
观察还款 AUC。训练目标和监控指标可以不同：模型按损失学习，按 AUC 选择树数。

## 4. 与逻辑回归的预处理有什么不同？

本轮保留 NaN，不做均值填补和标准化。树主要按大小关系和阈值划分，通常不需要
像带 L2 的逻辑回归那样统一列尺度。但 LightGBM 仍需要学习分箱边界和缺失方向，
不能因此认为它没有任何需要在折内拟合的处理。

`use_missing=True` 让模型在分裂时学习缺失值进入哪一侧；这个方向可以随特征和
节点变化，不是全局“缺失永远去左边”。`zero_as_missing=False` 保留实际零值的含义。
本轮没有人工缺失指示列，测试全空的 7 列也保留，作为与逻辑回归一致的初始输入集合。
原生缺失处理不能自动解决训练/测试缺失率漂移。

## 5. 最关键的验证设计：三种角色

每个外层折有三种数据角色：

| 角色 | 来源 | 用途 |
| --- | --- | --- |
| 内部拟合集 | 拟合四折中的 85% | 第一阶段建立分箱、学习树 |
| 内部早停集 | 拟合四折中的 15% | 第一阶段监控 AUC、选择树数 |
| 外层验证折 | 留出的第五折 | 最终评价，不参与树数选择 |

核心划分：

```python
candidates = np.flatnonzero(fold_id != fold)
outer_valid = np.flatnonzero(fold_id == fold)
inner_fit, early_stop = train_test_split(
    candidates,
    test_size=0.15,
    stratify=labels[candidates],
    random_state=42 + fold,
)
```

`flatnonzero` 把布尔条件变成训练行位置；`stratify` 保持两个内部集合的标签比例。
保存前将内部行位置排序，方便顺序读取磁盘数据；排序不改变分到哪个集合。

不能用外层验证集一边挑最佳树数，一边把同一分数当作完全独立的最终评价。
本项目避免这种情况：内部早停集负责选择，外层验证集负责最终打分。

## 6. 第一阶段怎样选树数？

```python
inner_data = make_dataset(..., indices=inner_fit)
stop_data = make_dataset(..., indices=early_stop, reference=inner_data)

selected = lgb.train(
    params,
    inner_data,
    num_boost_round=600,
    valid_sets=[stop_data],
    valid_names=['inner_early_stop'],
    callbacks=[
        lgb.early_stopping(50, first_metric_only=True, min_delta=0.0001),
        lgb.record_evaluation(history),
    ],
)
selected_rounds = selected.best_iteration
```

上面 `...` 表示省略已有的参数，用于阅读流程；直接运行完整脚本。
`reference=inner_data` 要求早停集使用内部拟合集建立的分箱边界，不独立从早停数据学习边界。
最多训练 600 轮，连续 50 轮未达到至少 0.0001 的有效 AUC 改善则早停。
`best_iteration` 是选定树数；训练可能为了等待 50 轮而额外尝试一些树。
因此“选定树数”和“实际监控了多少轮”不是同一个数。

如果到 600 轮才结束，说明预算上限到了，不等于已证明继续加树不会改善。
本轮固定这些配置，不通过外层结果反复提高预算。

## 7. 为什么还要第二阶段重训？

树数选好后，第一阶段模型只学过内部拟合集。为了让最终模型使用全部拟合四折：

```python
all_fit = np.flatnonzero(fold_id != fold)
full_data = make_dataset(..., indices=all_fit)
model = lgb.train(params, full_data, num_boost_round=selected_rounds)

oof[outer_valid] = predict_rows(model, values, indices=outer_valid)
```

这是从头重训，不是在第一阶段模型上继续追加树。
分箱边界也重新从全部拟合四折建立，仍没有使用外层验证或测试数据。
第二阶段固定树数，外层验证不传给 `lgb.train`，不再早停。
由于训练样本变化，第二阶段模型与第一阶段模型的树结构可以不同。

五折结束，每条记录由没有用过它的模型生成一次 OOF。保存的五折第二阶段模型
分别预测测试集，对还款概率求平均；没有额外全量重训第六个模型。

## 8. 参数逐项理解

| 参数 | 本轮取值 | 作用 |
| --- | --- | --- |
| `learning_rate` | 0.05 | 每棵新树对总分数的更新力度 |
| `num_leaves` | 31 | 每棵树最多 31 个叶子 |
| `max_depth` | -1 | 不额外限制深度，仍受叶子数等约束 |
| `min_data_in_leaf` | 200 | 约束小叶子，减少少量样本上的过拟合 |
| `lambda_l2` | 1.0 | 约束叶子更新值，稳定模型 |
| `feature_fraction` | 0.8 | 每轮随机使用约 80% 特征建树 |
| `bagging_fraction` | 0.8 | 每轮从拟合行中随机采样约 80% |
| `bagging_freq` | 1 | 每轮重新行采样，使上述参数生效 |
| `max_bin` | 63 | 每列分箱数量上限，控制效率与分裂精细度 |
| `bin_construct_sample_cnt` | 20,000 | 用拟合部分的样本估计分箱边界 |
| `deterministic`、固定 seeds | True、42+fold | 提高同一环境运行的可复现性 |

这些是固定基线配置，没有证明最优。`max_bin` 的直方图分箱不是按标签分组，
也不是删掉其余训练行；边界估计后，全体相应拟合行仍进入 Dataset。
每轮行采样属于模型训练策略，与五折验证划分、内存分批读取都不同。

## 9. 内存与数据接口

直接 `values[all_fit]` 可能产生约 2.4 GB 的新副本。`IndexedSequence` 只向 LightGBM
提供当前需要的行或 4,096 行批次，复用上一轮磁盘原始特征缓存。
接口需要 float64，因此只转换当前批；不把整个 float32 缓存一次扩成 float64。

LightGBM 会在内存保存分箱后的数据和直方图，设置 63 个 bin、128 MB 的直方图缓存
预算、4 个 CPU 线程，降低资源压力。分批读取不是把每批当成独立模型，也不是增量 SGD。

## 10. 怎样理解结果与重要性？

整体与压力组继续报告还款 ROC-AUC、违约 AP、log loss、Brier；准确率只作辅助。
AP 使用 `average_precision_score(1-label, 1-p_repay)`，不是固定阈值下的召回率。
比较代码先核对上一轮 OOF 的 id、标签、折号和压力标记，再重新计算全部模型指标。

重要性保存两种：gain 是该特征分裂贡献的目标改善，split count 是使用次数。
每折 gain 归一化为占比，再取五折均值。gain 不表示正负方向，不是业务因果，
也不等于它在测试集一定有用；相关特征可能分摊重要性。
本轮不根据重要性删列。漂移与缺失指示等处理留给独立的后续消融实验。

## 11. 核查、复现与面试表达

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\scripts\run_lightgbm_baseline.py
.\.venv\Scripts\python.exe .\scripts\verify_lightgbm_baseline.py
.\.venv\Scripts\python.exe .\scripts\plot_lightgbm_baseline.py
```

测试核对分批和普通数组训练的一致性、NaN/零值保留、内部/外层划分不交叉与保存重载。
最终核查重载五折模型复现 OOF 和测试概率，对照指标和提交模板，并核对分组与折表未变。
`--resume` 可以在相同来源、配置下复用完整折 checkpoint，不重跑已经成功的折。
绘图脚本生成内部早停 AUC 曲线和全量 OOF ROC/PR 对比图；指标使用全部预测，
曲线只对渲染顶点做抽稀。内部曲线不是外层验证曲线。

面试可以这样讲，具体指标补充实际结果报告中的数值：

> 我在固定分层五折上建立了 LightGBM 基线，使用全部匿名特征，保留原始 NaN。
> 与逻辑回归相比，提升树可以通过阈值分裂学习非线性和特征交互。为了保持外层
> 验证独立，我从拟合四折内部留出分层早停集选择树数，再用完整拟合四折重训，
> 最后生成外层 OOF。我使用磁盘缓存和 Sequence 接口构建分箱数据，控制宽表
> 内存需求，并报告整体、压力组和概率误差，与同一套 OOF 的逻辑回归比较。
> 测试提交来自五折重训模型概率均值；训练/测试漂移仍限制本地指标外推。
