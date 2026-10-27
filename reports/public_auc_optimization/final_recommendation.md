# 本轮 AUC 优化结果与提交顺序

更新（2026-10-25）：第一份三模型融合已提交，公榜 AUC 为 **0.73949**、排名第二。
用户截图中第一名为 0.74111，差距 0.00162，其余候选尚未提交。
下文保留首次训练结束时的推荐与核查记录；已提交文件不用重复上传。
下面的 AUC 都是开发用五折 OOF 成绩，不能直接换算成官方公榜成绩。

## 首次训练结束时推荐的文件

第一份已取得上述公榜成绩；后续提交应结合新反馈及下一轮训练结果。

| 顺序 | 候选 | OOF AUC | feature_492 高尾组 AUC | feature_1039 非缺失组 AUC |
|---|---|---:|---:|---:|
| 1 | LightGBM 50% + CatBoost 30% + XGBoost 20% | 0.821836 | 0.770374 | 0.787441 |
| 2 | LightGBM 70% + CatBoost 30% | 0.821132 | 0.769177 | 0.786511 |
| 3 | 删除四个漂移列的三种子 LightGBM 集成 | 0.820032 | 0.767429 | 0.785093 |

对应文件：

1. [三模型概率融合](D:/large-scale-loan-repayment-modeling-challenge/submissions/public_auc/blend_lgbm50_catboost30_xgb20_1db9ccc3c6.csv)
2. [LightGBM / CatBoost 概率融合](D:/large-scale-loan-repayment-modeling-challenge/submissions/public_auc/blend_drift_lgbm70_catboost30_fa80c9bb4c.csv)
3. [LightGBM 三种子集成](D:/large-scale-loan-repayment-modeling-challenge/submissions/public_auc/lgbm_drift_tuned_ensemble_87bcc2f428.csv)

第一份在五折中均优于第三份，整体提升约 **0.001804**，最低折 AUC 为 **0.820127**。
第二、三份使用删除四个主要漂移列的输入方案；第一份混入的历史 XGBoost 使用原始特征。
保留这些不同输入和模型组合，方便用实际公榜成绩判断这种差异的影响。
排序融合也是可用备选，但本轮三模型排序融合的 OOF AUC 为 0.821784，
没有优于对应的概率融合，因此没有占据前三个推荐位置。

## 实际完成的训练

- LightGBM：种子 42、2026、3407，各自完整五折，共 15 个最终模型。
- GPU CatBoost：深度 6、最多 1,200 轮，完整五折；整体 OOF AUC 为 0.815187。
- GPU CatBoost 容量对照：深度 8、最多 2,000 轮、L2=8，完整五折；整体 OOF AUC 为 0.818757。
- LightGBM 缺失特征筛选对照：完整五折；整体 OOF AUC 为 0.817919。
- XGBoost：复用上一阶段完成的五折预测，没有再次投入每折约 40 分钟的训练成本。

CatBoost 使用本机 RTX 3060 Laptop GPU。量化只在拟合行上建立边界，
验证行复用拟合边界；先保存并重载二进制 Pool，再释放原始数值引用。
深度 8 配方复用相同输入和量化缓存，配置与模型产物单独保存。
内部早停选树数后，在完整拟合四折上重训；外层验证折保持隔离。
模型继续排除 id，保留真实零和 NaN，不添加匿名特征的虚构业务含义。

训练默认单进程；可选多进程总预算为 10 个逻辑 CPU。
最后一轮实际并行为两个 CPU 工作进程各 4 线程，GPU 工作进程最多使用 2 个逻辑 CPU。
所有训练进程均已退出。本轮总训练安排保持在用户选择的约 1–2 小时内。

## 核查结果

- 本轮 30 个最终模型已重载，每个模型抽样核对 512 条 OOF 和 512 条测试记录，并重算完整保存 OOF 的指标；抽样预测最大差异为 0。
- 最终 28 份当前候选 CSV 的全部 36,324 行已独立重读，核对列名、唯一 ID、模板顺序、数值范围和文件哈希；按来源和权重重算全部测试分数，最大差异为 0。
- 核对 25 个不同来源文件的哈希，并重算所有候选的整体 OOF AUC。
- 8 个针对性测试通过，覆盖默认单进程、10 核预算、断点复用、ID 对齐、排序融合、提交版本保留及概率方向反转的检测。

证据：

- [完整候选指标](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/candidate_metrics.csv)
- [逐折指标](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/candidate_fold_metrics.csv)
- [提交文件核查](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/candidate_file_verification.json)
- [CPU 模型重载核查](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/model_reload_verification.json)
- [容量 CatBoost 模型重载核查](D:/large-scale-loan-repayment-modeling-challenge/reports/catboost_gpu_capacity/model_reload_verification.json)
- [操作说明与复现命令](D:/large-scale-loan-repayment-modeling-challenge/docs/public_auc_optimization.md)

提交后通过 `build_public_auc_candidates.py` 的 `--record-submission` 和 `--public-score`
登记实际返回成绩。文件名包含内容哈希，已有版本会保留，避免把旧成绩归到新的预测文件。
