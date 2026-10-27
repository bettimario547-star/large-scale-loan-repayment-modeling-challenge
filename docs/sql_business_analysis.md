# SQL 业务分析层：复现与解释

本阶段沿用已完成的模型，完成模型冻结、风险分层、固定阈值描述和分数分布诊断。
Champion 是原始 LightGBM 基线；Challenger 是调参 LightGBM 三种子集成。
没有继续训练 XGBoost/CatBoost，没有选择最终审批策略，也没有开展随机实验。

## 复现

在项目根目录 PowerShell 执行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_sql_business_analysis tests.test_id_feature_contract -v
.\.venv\Scripts\python.exe scripts/run_sql_business_analysis.py
.\.venv\Scripts\python.exe scripts/verify_sql_business_analysis.py
```

仅冻结产物可加 `--freeze-only`。重复运行默认复用相同冻结快照，重建本阶段独立数据库与报告。
若模型输入、质量数据库或阈值配置改变，旧快照拒绝复用；使用 `--run-name sql_business_analysis_v2` 创建新版本。
核查脚本也需传入相同的 `--run-name`。不要让两个进程同时写同一实验目录。

不需要原始 CSV，也不调用 LightGBM、XGBoost 或 CatBoost：运行读取现有 Parquet、折预测 checkpoint、模型文件及历史核查记录。
历史元数据中的旧机器路径保持原样，新冻结清单使用项目相对路径并记录当前文件 SHA256。
当前模型校验通过不等于重新推理通过：本阶段验证模型文件哈希、checkpoint 重建和数据对齐，
原生模型重载推理的证据来自保留的历史核查。本阶段没有重新读取 1,105 列特征做模型推理。

## 运行顺序

1. `freeze_models()` 检查数据 ID、标签、五折和历史审计状态；核对 20 个模型文件、重建 OOF 与测试均值。
2. 保存统一长表与模型登记表，写入冻结清单；冻结清单是最后写入的完成标记。
3. 在独立 DuckDB 中按顺序执行 `sql/03_analysis/01`–`05`；只读复用原库的物理质量统计与漂移表。
4. 导出表，用 NumPy/sklearn 独立重算，绘图并生成报告。示例查询在 `06_example_queries.sql`。

配置在 `configs/sql_business_analysis.json`：包括固定模型、13 个预声明阈值、4 线程及 1GB DuckDB 内存预算。
Python/pandas 和模型预测快照也会使用内存，DuckDB 配置并非整个进程的内存硬上限。

## SQL 在做什么

`01_model_predictions.sql` 建立 `(dataset, id, model_version)` 主键与概率/标签约束。
训练和测试 ID 都从 0 开始，不允许只用 ID 把两份数据拼接。

`02_model_performance.sql` 先把相同违约分数聚合，再按分数从高到低计算累计人数。
AUC 对同分正负样本记 0.5；AP 按每一同分组新增违约召回乘累计精确率；KS 取两类累计分布的最大距离。
这种实现与 sklearn 独立对照，常数预测不会被 ID 排序制造成有识别能力的模型。

`03_risk_bands.sql` 只使用训练 OOF 的 Q10–Q90 定义 D1–D10，然后套用到测试集。
同分保留在同一风险层，所以不保证每层恰好 10%；即使风险层为空，仍输出一行。
`default_lift` 是相对全体训练违约率的倍数，`repayment_lift` 是相对全体训练还款率的倍数。
累计违约捕获率从 D10 向 D1 累加，不代表真实拒绝行为已经减少了违约。

`04_approval_thresholds.sql` 用 `p_repay >= threshold` 描述阈值下的审批占比与获批组合表现。
阈值事先列在配置中；当前不从网格里选“最优阈值”，没有本金、利息、LGD 和收益。
Champion/Challenger 配对表使用相同概率阈值，两套模型审批率可能不同。

`05_prediction_distribution.sql` 输出分数统计与使用共用边界的 PSI、总变差距离。
OOF 与测试的平均模型数量不同，分数变化不仅可能来自数据漂移，也可能来自集成方式。
不能据此估算测试真实违约率或宣称标签分布漂移已经被修复。

## 必须保留的解释边界

- `1` 是还款、`0` 是违约；`p_default = 1-p_repay`。
- 训练业务评价只使用 OOF；已有 OOF 被多轮开发使用，因此本阶段是描述性开发分析。
- 测试没有标签，所有实际还款率/违约率、违约人数、提升度和捕获率保持 NULL，不能补成 0。
- 原始概率未校准；“平均模型违约分数”与“实际违约率”分别命名和展示。
- 无人获批时人数为 0，获批实际违约率为 NULL，不能报告成零风险。
- 未提供放款日期、还款观察窗口和客户稳定标识，不能声称时间验证或真实客户随机分流已经完成。
- 没有虚构匿名特征的业务含义，也没有将开发分数写成 Kaggle 官方成绩。

## 输出

数据库：`data/interim/sql_business_analysis.duckdb`。

冻结产物：`artifacts/sql_business_analysis/` 的 `frozen_predictions.parquet`、`model_registry.parquet` 和 `model_freeze_manifest.json`。

报告：`reports/sql_business_analysis/sql_business_analysis_report.md`，配套两张图、11 份 CSV、
`summary.json`、`run_config.json` 和 `artifact_verification.json`。字段字典见 `docs/sql_business_data_dictionary.md`。

下一阶段再确定经济假设、相同审批率比较和风险约束下的最终策略，然后设计 A/B Test。
