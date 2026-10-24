# 第二步：`id` 只负责对齐，不进入还款模型

这一页对应 [实际数据核查报告](../reports/id_role/id_role_report.md) 和
[`scripts/id_feature_contract.py`](../scripts/id_feature_contract.py)。本阶段尚未训练还款模型。

## 1. 先分清三个角色

想象一张贷款表：`id` 像行的“快递单号”，便于找到这一行；`feature_*`
是模型能观察到的贷款信息；`label` 是已知结果（本比赛中 **1=还款，0=违约**）。
预测时要用前者中的特征去学习结果，不能把答案本身塞给模型。

| 列 | 作用 | 进入模型的 X 吗？ |
|---|---|---|
| `feature_1`–`feature_1105` | 匿名预测特征 | 是 |
| `label` | 训练目标 y | 否 |
| `id` | 文件内部定位、连接折号、连接提交模板 | 否 |
| `fold`、压力组标记 | 验证安排与诊断 | 否 |

`id` 在本比赛不是经证实的放款时间，也不是贷款人的稳定业务编号。真实数据里，
训练集 id 是 0–688997，测试集又从 0–36323 开始，因此测试集全部 36,324 个
id 数字都能在训练集里找到。**同号不代表同一笔贷款**；错误地按 id 跨训练、测试
连接，可能把训练标签误接给测试记录。即使只将 id 当数值特征，模型也可能学习到
人为编号规律，而官方测试集的编号范围与训练集不同。这里的排除是有根据的项目
约定，不是声称所有任务中的 ID 都绝对无用。

## 2. 第一层保护：严格列出模型特征

```python
feature_names = model_feature_columns(train.columns, test.columns)
X_train = select_model_features(train, feature_names)
y_train = train["label"]
X_test = select_model_features(test, feature_names)
```

逐行理解：

1. `model_feature_columns` 检查训练集是否恰好有 `id,label,feature_1…feature_1105`，
   测试集是否恰好有 `id,feature_1…feature_1105`。它返回同一顺序的 1,105 列。
   少列、多列或重复列会直接报错，防止训练与预测的列错位。
2. `select_model_features` 使用**白名单** `frame.loc[:, feature_names]` 取模型输入。
   它还检查传入名单必须恰好是那 1,105 列，所以即便后续 `train` 又连接了
   `fold` 等元数据，也不会误把它们放进 X。不要写 `train.drop(columns=["label"])`：
   这样会把 `id` 和后来加上的 `fold` 留在 X 中。
3. `y_train` 单独取 `label`。这是模型要学习的答案，不是预测时可用的特征。

上面是未来训练模型时的调用示意，不会在这一步运行全量建模。完整 `train` Parquet
较大（约 2.8 GB），内存不足时需要分批、按需读列或使用 DuckDB 投影。

## 3. 第二层保护：五折表按训练集 id 对齐

五折表不是模型特征，它告诉我们每一行被分到哪一折。假设训练数据按 id
`[12, 10, 11]` 排列，而折表按 `[10, 11, 12]` 排列。如果直接取折表的
`fold` 数组，就会给训练行贴错折号。我们用：

```python
folds_aligned = align_validation_folds(train["id"], folds)
fold_id = folds_aligned["fold"].to_numpy()
```

函数内部先要求两边 id 非空且各自唯一，再使用 `merge(..., on="id",
validate="one_to_one")` 做一对一连接；如果少一行或多一行就报错。最后按训练
数据的原行序返回折号。对上述例子，正确折号是 `[2, 0, 1]`，不是 `[0, 1, 2]`。
真实数据核查结果：688,998 个训练 id 与五折表 id 完全匹配，两边未匹配行数
都是 0，折号仅有 0–4。

SQL 中对应的想法是：

```sql
SELECT t.id, f.fold
FROM processed_train AS t
JOIN read_parquet('data/interim/validation_folds.parquet') AS f
  ON t.id = f.id;
```

这里的连接只在**训练集与其附属折表**之间成立，不能拿来连接训练集和测试集。

## 4. 第三层保护：提交时也按 id 对齐

模型以后应输出 `label=1` 的**还款概率**。先建立一张两列表：`id,p_repay`。
官方模板是 `id,prediction`，其中现有的 0.5 只是占位值。调用：

```python
test_predictions = pd.DataFrame({"id": test["id"], "p_repay": p_repay})
submission = build_submission(test_predictions, sample_template)
# 只有 p_repay 确实来自训练好的模型且完成检查后，才保存：
# submission.to_csv("submissions/model_v1.csv", index=False)
```

`build_submission` 不相信“第 N 个预测对应模板第 N 行”：它按 `id` 一对一
连接，保留模板顺序，检查 id 不重复、不缺失、集合完全一致，检查概率是有限数
且在 `[0,1]` 内。最终列名严格为 `id,prediction`。例如模板顺序为
`[2,0,1]`、预测顺序为 `[1,2,0]`，输出仍是模板顺序，并把各自正确的
概率带到对应 id。函数本身不会写 CSV，因此不会误产生一份 0.5 占位提交。

若用 scikit-learn 的 `predict_proba`，不要想当然地取第 2 列；应该检查
`model.classes_`，找到类别 `1` 对应的列，再赋给 `p_repay`。

## 5. 如何复现、面试怎么讲

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\scripts\audit_id_feature_policy.py
```

单元测试使用小型合成数据，故意打乱行序，并尝试重复/缺失 id、非法概率和
错误特征列；真实数据审计检查四张表的 id 唯一性及双向未匹配行数。项目没有
在本阶段生成提交文件或声称已经有还款模型成绩。

面试可以这样回答：

> 我先把数据列分成预测特征、训练目标和记录键三类。这个比赛的测试 id
> 从 0 重新编号，与训练 id 大量重合，因此不能把 id 当作跨数据集的贷款键，
> 更没有证据把它当作时间或业务变量。我用 `feature_1` 到 `feature_1105`
> 的显式白名单构造 X，确保 id、label、fold 不进入模型；再用 id 对齐训练
> 折号和测试提交模板，并用一对一连接、双向反连接及单元测试检查无漏行、
> 无重复、无错位。
