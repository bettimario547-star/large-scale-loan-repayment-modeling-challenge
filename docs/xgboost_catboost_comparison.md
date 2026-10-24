# 第三步：XGBoost / CatBoost 对照学习说明

对应实际训练代码 `scripts/run_tree_model_comparison.py`。
完整成绩、树数、逐折变化和核查结果由
`scripts/report_tree_model_comparison.py` 写入
`reports/tree_model_comparison/tree_model_comparison_report.md`。

## 0. 先用一个例子理解“梯度提升树”

假设一个客户最终还款，标签是 1，但模型目前只给出 0.3 的还款概率。
这次预测偏低，后续树会根据损失的梯度学习怎样修正这类样本的预测。
对于实际违约、却被预测成高还款概率的样本，修正方向则相反。
每一棵新树关注的是当前损失需要怎样改变，不是把标签重新从零预测一次。

多个模型的平均与提升树内部的树相加是两件事。单个提升树模型先把树的输出
累加到一个实数分数，再经过 sigmoid 得到概率：

```text
F_t(x) = F_(t-1)(x) + learning_rate * tree_t(x)
p_repay(x) = 1 / (1 + exp(-F_t(x)))
```

`F` 是未转换为概率的分数；树叶输出可以为正，也可以为负。
因此不能把它理解成“900 棵树分别给出概率，再简单平均”。
最终测试预测里的五折模型平均，才是在平均五个已经转换好的概率。

二分类 Logloss 可以写成：

```text
loss = -mean(y * log(p_repay) + (1-y) * log(1-p_repay))
```

如果真正还款却给出接近 0 的还款概率，损失会很大；真正违约却给出接近 1 的概率也一样。
这个函数对模型分数可求导，能指导建树。AUC 比较正负样本的排序，不能直接作为本轮
常规建树的可微训练目标，所以用 Logloss 训练、用内部 AUC 选择轮数。

面试若追问 XGBoost 的二阶信息：对二分类分数，梯度是 `p-y`，Hessian 是 `p*(1-p)`。
在只考虑 L2 正则的简化情况下，一个叶子的输出近似为 `-sum(gradient)/(sum(Hessian)+lambda)`。
这解释了 `lambda` 为什么会压小叶子输出，以及 `min_child_weight` 为什么约束 Hessian
之和而不是直接约束样本数。不必把这条简式当作所有库的完整实现。

## 1. 为什么已经有 LightGBM，还要比较两个模型

同样属于梯度提升树，不同库在树的生长、分裂搜索、约束和缺失值处理上有不同设计。
数据上适合哪一种，需要使用同一批验证样本比较。本轮同时检查排序、违约识别排序、
概率误差、逐折稳定性和训练成本，并保留不如现有模型的完整结果。

数据为 688,998 条训练记录、36,324 条测试记录、1,105 个匿名数值特征。
继续使用全部原始 float32 特征，保留 NaN 和真实零值。
标签 `1` 是还款，`0` 是违约。所有模型输出 `p_repay`，违约概率是 `1 - p_repay`。
`id` 只用来对齐记录和提交，训练/测试 id 的数值重叠不能用来连接标签。

上一轮特征筛选消融没有带来整体 AUC 优势，所以下一步调参和本轮对照均继续使用原始
1,105 列。这样比较中的特征方案保持一致，不会把改模型与改特征的效果混在一起。

## 2. 三个模型的共同点和本轮具体区别

三者都逐轮增加树，根据当前预测的损失修正模型。本轮都训练二分类概率，使用内部 AUC
选择树数。AUC 是评价指标，训练时使用的是二分类 logistic / Logloss 损失。

| 项目 | 本轮 LightGBM 参考 | 本轮 XGBoost | 本轮 CatBoost |
|---|---|---|---|
| 树的生长 | 优先分裂收益最大的叶子 | `depthwise`，按深度扩展 | `SymmetricTree`，同一层使用相同分裂条件 |
| 主要复杂度约束 | 叶子数、叶子样本约束 | 最大深度、叶子 Hessian 约束 | 对称树深度、叶子 L2 约束 |
| 数值缺失 | 学习缺失默认分支 | 学习缺失默认分支 | `nan_mode=Min`，缺失作为低于所有有限数值的独立状态 |
| 本轮类别特征 | 无显式类别列 | 无显式类别列 | 无显式类别列 |
| 本轮训练方式 | GBDT | CPU `hist` | CPU `Plain` |

CatBoost 库支持类别特征编码和 Ordered boosting，但这些不能自动作为本轮做过的工作。
匿名特征没有经过业务确认的类别含义，本轮全部按数值处理，明确选择 `boosting_type=Plain`。
对称树的一层所有节点使用同一个分裂条件；深度 6 的树最多有 64 个叶子。
XGBoost 的深度 6 树也最多有 64 个叶子，但各节点可以使用不同条件。
相同叶子上限不会让两种树的表达能力完全相同。

## 3. “公平对照”具体固定了什么

固定原有五折、内部 85%/15% 分层划分、全部原始特征、标签方向和评价指标。
每套新库提前指定一组配置、一个基础种子 42，实际每折种子为 `42 + fold`。
最多 900 轮，60 轮没有超过 0.0001 的有效内部 AUC 改善就停止，使用 CPU 4 线程。
候选配置不根据外层分数追加或修改。

比较表包含原 LightGBM 基线、调参后的 LightGBM 种子 42、三种子 LightGBM 集成，
以及本轮两个单种子新模型。单种子 LightGBM 与新模型都是五折最终模型；
已有 LightGBM 集成是 15 个最终模型，模型数量和预测成本应同时解释。
之前 LightGBM 比较了四组参数，本轮每个新库只有一组，因此参数搜索预算不同。
本轮是具体配置和训练流程的开发对照，不能据此证明某个算法在所有数据上更强。

`max_bin=63` 与 CatBoost `border_count=63` 只是各自库的量化配置。
一个是分箱数上限，一个是边界数量上限，量化算法也不同；不能称作完全相同的分箱。

## 4. 如何隔离外层验证

以第 0 折为例，约 13.78 万行留出；其余约 55.12 万行分成约 46.85 万行内部训练和
8.27 万行内部早停。只在内部训练行建立分箱并训练，只按内部早停 AUC 选择树数。
之后在完整拟合四折上重新建立分箱，从头训练所选树数，最后预测留出折。

```python
inner, stop, valid = base.split_outer_fit(labels, fold_id, fold, 0.15)
all_fit = np.flatnonzero(fold_id != fold)
```

`inner`、`stop`、`valid` 是行位置，分别对应内部训练、早停、外层验证。
`all_fit` 是完整其余四折。分层划分的种子固定为 `42 + fold`。
`base` 是复用的已有 LightGBM 基线模块，提供数据契约、划分、缓存和统一指标。

OOF 是 out-of-fold：每条训练记录由排除其所在折的模型预测。
本轮每个库各训练五个内部模型和五个完整四折重训模型，总计 20 个新模型。
只有完整四折重训模型生成最终 OOF/测试结果。

## 5. XGBoost 的参数和代码

```python
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 20.0,
    "lambda": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_bin": 63,
}
```

`binary:logistic` 输出还款概率；`hist` 把连续值量化后用直方图搜索分裂；
`eta` 是学习率；`max_depth` 限制深度；`lambda` 是叶子输出的 L2 约束；
行列抽样比例都是 0.8。`min_child_weight=20` 约束叶子内 Hessian 之和，
不是每个叶子至少 20 行，也不能直接等同于 LightGBM 的叶子样本数。

```python
train_data = xgb_data(values, labels, inner, names, threads, batch)
stop_data = xgb_data(values, labels, stop, names, threads, batch, ref=train_data)

callback = xgb.callback.EarlyStopping(
    rounds=60, min_delta=0.0001,
    metric_name="auc", data_name="inner_early_stop",
    maximize=True, save_best=True,
)

model = xgb.train(
    params, train_data, num_boost_round=900,
    evals=[(stop_data, "inner_early_stop")],
    callbacks=[callback],
)
rounds = int(model.best_iteration) + 1
```

`save_best=True` 保留按规则选择的模型。XGBoost 的 `best_iteration` 从 0 开始，
因此转换成树数需要加 1。完整四折重训时使用选出的 `rounds`，没有外层 `evals`。

## 6. XGBoost 如何节省内存

`BatchIter` 继承 `xgb.DataIter`，实现 `reset()` 和 `next(input_data)`。
每次从内存映射中只取一批行，转换成连续 float32 数组，再把特征、标签和名称交给库。
`QuantileDMatrix` 用它直接生成量化数据，避免完整原始拟合矩阵的额外复制。
原始缓存由 `BatchedFeatureFile` 访问：每批打开只读映射、复制所需行，然后立即关闭映射。
这样训练时不会同时长期保留原始矩阵映射和完整量化矩阵的驻留页。

```python
iterator = BatchIter(values, labels, indices, names, batch_rows)
matrix = xgb.QuantileDMatrix(
    iterator, missing=np.nan, max_bin=63, nthread=threads, ref=ref
)
```

内部早停矩阵的 `ref=train_data` 复用训练分箱，早停行不能决定量化边界。
复制出来的批次拥有自己的内存，因此关闭原始文件映射后，批次仍有效。
关闭映射只改变内存生命周期，不修改源缓存、数值、NaN 或记录顺序。
内存映射和分批读取并不代表零内存：库仍需要量化数据、分箱草图和直方图。
最终预测使用 `inplace_predict()` 逐批输入原始特征，输出概率。

## 7. CatBoost 参数如何理解

```python
params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "boosting_type": "Plain",
    "grow_policy": "SymmetricTree",
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "rsm": 0.8,
    "border_count": 63,
    "nan_mode": "Min",
    "use_best_model": False,
}
```

`depth=6` 控制对称树深度；`l2_leaf_reg` 约束叶子输出；
`Bernoulli` 配合 `subsample=0.8` 抽取训练行；`rsm` 控制分裂选择时的特征抽样。
`random_strength=1.0` 控制分裂评分的随机扰动，
`boost_from_average=True` 用拟合标签的平均水平初始化二分类预测，
`leaf_estimation_iterations=10` 每棵树做十次叶子值估计更新。

CatBoost 的行列抽样、L2 和分裂评分实现与另两套库不同，数值相同不等于效果相同。
`nan_mode=Min` 保留缺失状态，库可以考虑把缺失与有限值分开的分裂。
真实数值零始终是零。

## 8. CatBoost 如何处理大矩阵和分箱隔离

完整原始拟合矩阵较大，本轮先分批导出仅含标签和数值特征的临时 TSV。
使用 `catboost.utils.quantize()` 从文件直接构造量化 Pool，官方接口会避免把完整未量化
原始矩阵一次载入内存。量化后删除本轮创建的精确临时文件，保留边界文件用于核查。

```python
train_pool = quantize(
    fitting_tsv, column_description=columns_cd,
    border_count=63, nan_mode="Min", task_type="CPU",
)
train_pool.save_quantization_borders(training_borders)

stop_pool = quantize(
    early_stop_tsv, column_description=columns_cd,
    input_borders=training_borders,
    ignored_features=features_without_training_borders,
)
```

代码输出 9 位有效数字，以便 float32 数值往返不变；`nan` 保留缺失。
列说明文件写明第 0 列是标签，其余列都是 Num，并保留原特征名。

如果某列在内部训练中恒定或全缺失，没有可用边界，而在早停数据里发生变化，
也不能让早停数据为它新建边界。因此早停量化显式忽略训练没有边界的列，
并检查保存出的早停边界文件与训练边界完全相同。
这是训练阶段自然产生的无可用分裂列处理，没有根据外层数据做人工特征筛选。

完整四折重训的边界重新由完整四折估计，测试与外层验证不参与。
`used_ram_limit=3gb` 是库的资源请求，不是整个 Python 进程的硬内存上限。
为降低内存竞争，本轮使用一个训练进程顺序运行。

## 9. CatBoost 早停为什么写了回调

为了和已有流程使用同一条“改善超过 0.0001、耐心 60 轮”规则，
本轮用 CPU Python 回调 `CatStop.after_iteration()` 读取内部验证 AUC。
`AUCStop` 保存认可的最佳轮数，达到耐心后返回 False 停止。

```python
callback = CatStop(60, 0.0001)
model = cb.CatBoostClassifier(**params, iterations=900)
model.fit(train_pool, eval_set=stop_pool, callbacks=[callback], verbose=False)

rounds = callback.state.best_round
model.shrink(rounds)
```

`use_best_model=False` 避免库按它自己的数值最大值规则另选一次。
`shrink(rounds)` 保留我们规则选中的前 rounds 棵树。
这个轮数也用于完整四折从头重训；重训没有验证数据或早停。
预测时 `predict_proba(pool)[:, 1]` 取类别 1（还款）概率。

## 10. 汇总、评价和概率方向

每个库的五折预测填回原行位置，检查每行覆盖一次，然后计算整个 OOF 指标。
测试预测等权平均五个完整四折模型，不在本轮搜索跨模型混合权重。

```python
oof[valid_index] = p_repay_valid
test_prediction += p_repay_test / 5

roc_auc_score(labels, oof)
average_precision_score(1 - labels, 1 - oof)
log_loss(labels, oof, labels=[0, 1])
brier_score_loss(labels, oof)
```

AUC 评价排序；违约 AP 特别关注约 8.33% 的违约少数类；log loss 和 Brier 评价概率误差。
AP 使用 average precision，不是梯形积分的 PR 曲线面积。
总 OOF AUC 不能简单用五折 AUC 均值替代。

逐折差值在同一验证行上与调参 LightGBM 单种子和集成分别比较。
既有两个压力组仍报告四项指标，其样本有重叠，不能相加。
预测相关性用于观察不同模型是否产生不同排序/概率，不能直接等同于错误相关性，
也不能单凭相关性低就断言混合一定有效。

## 11. 如何选择下一步开发候选

提前固定：新模型相对已有 LightGBM 三种子集成，AUC 至少提高 0.0005，
违约 AP 下降不超过 0.001，log loss 增加不超过 0.0005，才考虑替换。
有多个符合时选择 AUC 较高者，否则继续使用已有集成。
这是开发工程规则，不是统计显著性检验或业务部署审批。
达到 900 轮上限的次数必须报告；用完预算不能称为已经找到最优树数。

## 12. 复现命令和文件

在项目根目录 PowerShell 中依次执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_tree_model_comparison.py --stage init
.\.venv\Scripts\python.exe -u scripts/run_tree_model_comparison.py --stage train
.\.venv\Scripts\python.exe scripts/run_tree_model_comparison.py --stage collect
.\.venv\Scripts\python.exe scripts/verify_tree_model_comparison.py
.\.venv\Scripts\python.exe scripts/report_tree_model_comparison.py
```

训练可以指定 `--engines xgboost` 或 `--engines catboost`，也可以指定 `--folds 0 1`。
不同时启动多个大矩阵量化任务；两个库五折全部完成后再 collect。
已完成模型校验签名、文件 SHA256 后自动复用，参数或代码变化时拒绝混用。
预算、线程和批大小在 init 固定，train 从已签名配置读取。

模型目录 `artifacts/tree_model_comparison/inner_models`、`models`；
分割与概率目录 `inner_splits`、`inner_predictions`、`fold_predictions`；
CatBoost 边界目录 `borders`。最终保存 OOF 与测试 Parquet。
两个本地提交为 `submissions/xgboost_baseline.csv` 和 `submissions/catboost_baseline.csv`。
报告包括整体比较、逐折比较、树数、耗时、相关性和核查结果。

训练时可在另一个终端运行只读进度脚本，不会加载完整特征或重新训练：

```powershell
.\.venv\Scripts\python.exe scripts/show_tree_model_progress.py
```

它只报告已经保存的完整阶段和单折结果。尚未集齐五折时，不能把部分折成绩当成全量 OOF。

## 13. 核查在证明什么

测试直接调用原生库，检查 TSV 数值往返、训练/早停量化边界一致、停止规则，
以及修改外层特征与标签后选中的树数和固定探针预测不变。
验证脚本重载 10 个内部模型和 10 个最终模型，重新生成所有保存概率，
核查分割无交叉、参数记录、特征名、树数、校验值、OOF 覆盖、指标和提交对齐。
重载一致证明实验可追溯，不代表未来数据表现已经得到保证。

## 14. 面试回答与结果边界

**为什么做 XGBoost/CatBoost 对照？**

我想检查已有 LightGBM 选择是否依赖单一库，所以使用同一五折和原始特征，
比较不同树生长方式下的排序与概率质量，同时观察成本。

**CatBoost 在你项目里用了哪些优势？**

本轮使用数值缺失处理和对称树训练，没有把匿名数值列猜成类别列。
明确使用 Plain boosting，不能声称使用了类别目标编码或 Ordered boosting。

**你怎么避免用验证集调出虚高分数？**

外层验证折被留出，其余四折内部选择树数，分箱只用拟合行。
选择后完整四折从头重训再评价。此前项目已多次使用这套折，因此最终报告开发 OOF，
仍需要新的保留数据或官方测试评价。

**某个模型分数低说明什么？**

说明这组预先指定配置和预算下的开发结果较低。
不同库的参数语义、搜索预算不同，不能证明整个算法一定不适合。
如果达到轮数上限，还应明确它可能受到预算限制。

测试 feature_492 的未见范围漂移仍然存在；float32 舍入不会因文本导出或转为 float64 恢复。
本轮没有官方测试成绩、概率校准、阈值成本优化、跨库集成或生产部署。
简历应根据完成报告填写真实 OOF 数字和得到的模型选择结论，不能将开发分数写成官方成绩。

## 15. 如何按函数读懂实际脚本

建议先看主流程与 `run_fold()`，理解数据如何流动，再看两套库的数据接口。
不要从文件第一行开始死记所有工具函数。

| 函数/类 | 输入与作用 | 为什么需要它 |
|---|---|---|
| `initialize()` | 固定配置、版本、数据与参考产物签名 | 开始前写清楚实验规则，避免跑到一半改口径 |
| `checked_config()` | 核对当前代码与参考文件是否符合签名 | 中断恢复不能混用另一套实验的结果 |
| `load_context()` | 加载标签、折号、特征文件和 LightGBM 参考概率 | 三个库沿用相同样本、顺序与比较对象 |
| `params_for()` | 按库与折号返回固定参数 | 参数在代码中可查，种子固定为 42+fold |
| `run_fold()` | 内部选轮数、完整四折重训、预测与保存 | 是本轮单折实验的核心 |
| `BatchIter` / `xgb_data()` | 分批提供 XGBoost 量化矩阵 | 控制完整原始矩阵额外复制 |
| `BatchedFeatureFile` | 一批取完即关闭原始只读映射 | 降低原始缓存与量化数据同时驻留的内存压力 |
| `export_numeric_tsv()` / `cat_data()` | 导出临时数值文件、从文件量化 CatBoost Pool | 适配本机资源，并保留 NaN、零与量化隔离 |
| `CatStop` / `AUCStop` | 读取内部 AUC，执行认可改善与耐心规则 | 和 XGBoost 使用同一条轮数选择规则 |
| `predict_rows()` | 分批预测指定行，检查概率范围 | 大数据预测也需要控制内存，且要保持行顺序 |
| `save_checkpoint()` / `checkpoint()` | 保存与核对模型、预测、参数和校验值 | 已完成阶段可安全复用，也便于定位结果来源 |
| `collect()` | 五折 OOF 拼回原行，测试概率平均，写比较指标 | 训练结束后才计算完整结果与开发推荐 |

`run_fold()` 里的 `inner`、`stop`、`valid` 都是 NumPy 行索引数组，不是三份永久复制的大矩阵。
`values[rows]` 才表示读取所需样本。`np.flatnonzero(fold_id != fold)` 找出不属于外层留出折的
所有行位置。例如正在评价第 0 折时，折号为 1、2、3、4 的行进入 `all_fit`。

```python
if selection is None:
    # 这个内部阶段没有完整 checkpoint，才训练内部模型并选择轮数。
    ...
if record is not None:
    # 完整四折模型及预测已保存、且校验通过，直接复用。
    return record
```

这两个分支解释了为什么中断恢复不必重新跑已经完成的折。校验不仅检查“文件在不在”，
还核对实验签名及模型/预测的 SHA256，防止把旧模型当成本轮结果。

在 `collect()` 中，OOF 是按各折留出行的位置拼回去，测试预测是平均五个模型：

```python
# 这条训练记录只能被没有训练过它的那一个最终折模型评价。
oof[valid_index] = p_repay_valid

# 测试记录无标签、也不属于训练五折，可以由五个最终模型共同预测。
test_prediction += p_repay_test / 5
```

不能把训练行也平均五个模型后拿来报告 OOF，因为其中四个最终模型已经见过那条训练行。
本轮主指标是在全部 688,998 条 OOF 概率上重新计算，不能把五个折 AUC 简单平均后代替。

`predict_proba(... )[:, 1]` 中，冒号表示取全部行，1 表示取第二列。
分类器的两列分别是类别 0 和类别 1 的概率；本轮重载核查了这个顺序，第二列才是还款概率。
`np.savez_compressed()` 保存行索引与概率数组；Parquet 保存汇总表；UBJ/CBM 保存可重新加载的树模型。
这些文件各有职责，保存一份提交 CSV 不能替代完整的实验记录。

## 16. 自己练习时检查是否理解了

1. 不看说明，指出 `run_fold()` 中第一次训练、轮数选择、第二次训练及外层预测分别在哪里。
2. 用标签 0 和还款概率 0.9 的例子，解释为什么违约 AP 要用 `1-label` 与 `1-p_repay`。
3. 解释为什么测试集平均五个模型，而 OOF 每行只能采用一个模型。
4. 解释为什么没有填补 NaN，也没有标准化：本轮树模型有原生缺失处理，阈值分裂通常不需要标准化；
   原始缺失和真实零值必须保持不同状态。
5. 解释内部早停改善超过 0.0001 才被认可时，保留轮数未必是曲线上数值最高的那一轮。
6. 说明这轮可以支持怎样的模型选择结论，以及为什么仍需要官方测试或新的保留数据。
