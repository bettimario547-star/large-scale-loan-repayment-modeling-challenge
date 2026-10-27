# 公榜收尾：最后一次提交

当前最高为 **0.74065**，来自 CatBoost25% + 原XGBoost75%。

| 原XGBoost权重 | 实际公榜 AUC |
| --- | --- |
| 25% | 0.73892 |
| 50% | 0.74040 |
| 75% | 0.74065 |

最后建议提交现有的 [纯原XGBoost五折均值](D:/large-scale-loan-repayment-modeling-challenge/submissions/public_auc/existing_xgboost_f133b11c17.csv)。

这是已存在的100%原XGBoost版本，没有新增训练、生成新模型或新融合。仅凭当前权重走势不能确定100%会继续提高。
36,324行ID、概率方向、分数范围、来源SHA256，以及五折测试预测等权均值均重新核查通过。

如果未超过0.74065，保留已提交的01c3571821版本作为首选。截图榜首0.74111，差距0.00046；最新名次未实时查询。

## 历史训练与核查

## 完成的实验

| 实验 | 完成范围 | OOF AUC |
| --- | --- | --- |
| 深度8 CatBoost 新种子2026 | 五折 | 0.817953 |
| 深度8 CatBoost 新种子3407 | 五折 | 0.818515 |
| CatBoost 种子42/2026/3407等权集成 | 三个完整五折种子 | 0.819668 |
| 深度9 CatBoost，L2=12 | 五折 | 0.819241 |
| LightGBM 255分箱 | 五折 | 0.817963 |
| 删四个漂移列的固定600轮XGBoost | 五折 | 0.818405 |

CPU 训练采用 8 线程 XGBoost/LightGBM 与最多 2 个逻辑 CPU 的 GPU 工作进程，
总训练预算 10。两项 GPU 训练顺序执行；多种子末尾的 CPU 导出期间先运行了深度9第一折。
沿用相同外层五折，早停使用四折拟合数据内部划分；新 XGBoost 的固定树数不使用外层验证早停。
排除 id，保留 NaN 与真实零，所有新提交均来自完整五折。

## 核查与证据

新增 25 个折模型重载成功，最大抽样预测差为 0。
独立检查 56 份 CSV、37 个来源文件，
每份均有 36,324 行，核查了 ID 行序、还款方向、所有分数、完整 OOF AUC 和来源哈希。
三种子集成的完整 OOF 与测试预测已独立重算均值。

- [本轮精确基准与推荐清单](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/round2_completion_summary.json)
- [与真实公榜最佳文件的逐折及压力组比较](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/round2_candidate_comparison.csv)
- [独立 CSV 核查](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/candidate_file_verification.json)
- [多种子真实 GPU 小样本测试](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/multiseed_gpu_smoke_test.json)
- [固定轮数 XGBoost 真实子进程测试](D:/large-scale-loan-repayment-modeling-challenge/reports/public_auc_optimization/xgboost_fast_smoke_test.json)

## 重新生成与核查

```powershell
.\.venv\Scripts\python.exe scripts/build_public_auc_candidates.py
.\.venv\Scripts\python.exe scripts/verify_public_auc_candidates.py
.\.venv\Scripts\python.exe scripts/report_public_auc_round2.py
```

粗粒度 25:75、50:50、75:25 权重对照已生成，包含原 0.74009 组件的对照；
后续按实际公榜反馈选择。提交文件按内容哈希保留版本，队列排除已登记成绩的确切文件。
