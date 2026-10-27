# 复现指南 / Reproduction guide

[交付导航](INDEX.md) · [架构](architecture.md) · [事实快照](project_facts.json) · [环境快照](environment_snapshot.json)

统一入口为 [scripts/project.py](../scripts/project.py)。所有命令从项目根目录运行。默认 `status`、`doctor`、文档核查和 `--dry-run` 都不训练、不改数据。业务核查调用只读函数，不调用会重写历史审计 JSON 的旧 CLI 包装。

## 两种审阅环境

| 环境 | 可完成 | 不应声称 |
| --- | --- | --- |
| 仅源码/聚合包 | README/卡片/配置/SQL/架构、事实快照、PBIP 聚合展示；Python 标准库文档核查 | 已重算不可用的 Kaggle 原数据或模型；源哈希不可用时会明确列出 |
| 当前完整本地工作区 | Parquet、折号、模型、冻结业务数据库/行级结果及只读复算 | 已从原始 CSV 重建；当前缺 `train.csv`、`test.csv` |

目录忽略大型 CSV/Parquet/DuckDB、模型和 submissions；23 张小型 Power BI 聚合 CSV 是例外。报告 Markdown/JSON、事实/环境快照和代码可保留。仅源码副本中缺少的私有证据链接会标注为不可用，关键文档和已交付的源码链接缺失会判失败。

## 环境与首次检查

文档审阅需要 Python 3.11+，不依赖 DuckDB 或模型包。当前执行环境是 Windows 11 / Python 3.13.11；Power BI Desktop 2.159.883.0。

```powershell
python scripts/project.py status
python scripts/project.py doctor --profile docs
python scripts/project.py verify --scope docs
python scripts/project.py test --scope delivery
```

完整依赖安装在新工作副本的虚拟环境中：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/project.py doctor --profile analysis
.\.venv\Scripts\python.exe scripts/project.py doctor --profile full
```

`docs` 检查文档；`analysis` 检查数据分析依赖、Parquet/折号/两套 OOF；`full` 再检查原始 CSV 和树模型包。退出码 0 表示检查通过，2 表示缺前置项或交付失败。`doctor` 只检查路径/包版本，不保证数据内容或深度统计复算通过。

[requirements.txt](../requirements.txt) 是兼容范围，[requirements-lock.txt](../requirements-lock.txt) 是当前 Windows 审阅环境的全部安装包快照，可在同类新环境用 `pip install -r requirements-lock.txt` 尝试还原。锁文件包含 Windows 包，不能原样用于所有平台。历史冻结模型记录 LightGBM 4.7.0 / NumPy 2.5.3；当前审阅 NumPy 2.5.4。模型签名可能因此拒绝复用；跨环境不保证模型字节或末位浮点一致，不能通过删除签名规避核查。

reportlab/pypdf 仅用于重新生成一页 PDF；当前 `.venv` 未安装它们，已有 PDF 来自独立文档运行时。本次文档交付不重建已冻结 PDF，也不安装依赖或重训练。

## 单一入口与验证范围

| 命令 | 行为 |
| --- | --- |
| `status` | 读取事实快照，显示保留 Champion、收益区间、样本与 Desktop 验收范围 |
| `doctor --profile docs/analysis/full` | 检查选定环境的路径/依赖，不扫描全部数据 |
| `verify --scope docs` | 文档/链接/关键口径/源哈希/架构 XML/入口完整性；可用私有源齐全时重新汇总事实 |
| `verify --scope docs --write` | 同上，仅写 `docs/delivery_verification.json`；不改冻结实验审计 |
| `verify --scope dashboard` | PBIP/schema/字段引用/聚合数据/空值及数学检查；不替代 Desktop 所有页面验收 |
| `verify --scope business --quick` | 只读核查五阶段：冻结哈希、行级/SQL、导出、规划/推断、已有校准规则与 Bootstrap；不重跑全部 Monte Carlo/哈希重复样本 |
| `verify --scope business` | 同上，加 A/A/V2 全部重复样本复算，可能耗时；从冻结输入重复模拟，不获得新业务样本 |
| `test --scope delivery` | 文档/事实篡改、链接失效、冻结保护、dry-run 与失败停止测试 |
| `test --scope business` | 交付测试 + 现有 SQL/策略/A/B/A/A/V2/Power BI 单元测试；测试夹具，不重训练模型 |
| `reproduce --target ... --dry-run` | 打印阶段计划、已有冻结状态及阻止重建的原因，不执行 |

`--quick` 仅适用于业务核查。非文档的 `--write` 不支持。各阶段命令失败立即停止，不继续下游阶段。文档核查通过仅表示明示的范围通过，历史统计审计仍以原冻结记录为证。

## 从零重建的阶段依赖

在**干净工作副本**中，从竞赛自行获取 `train.csv`、`test.csv`、`sample_submission.csv` 并放在根目录，遵守竞赛规则。运行前查看计划：

这里的干净副本保留源码、SQL、配置和文档，不复制已有 `data/`、`artifacts/`、生成的模型/业务 `reports/<stage>/` 输出。普通源码分享副本可能带历史报告供审阅；从零运行应另建输出为空的副本，避免入口将这些历史记录误当作待覆盖的工作输出。原工作区和历史报告继续保留。

```powershell
.\.venv\Scripts\python.exe scripts/project.py reproduce --target data --dry-run
.\.venv\Scripts\python.exe scripts/project.py reproduce --target models --dry-run
.\.venv\Scripts\python.exe scripts/project.py reproduce --target business --dry-run
.\.venv\Scripts\python.exe scripts/project.py reproduce --target dashboard --dry-run
```

去掉 `--dry-run` 才实际执行。每次只运行目标阶段，前置阶段不会隐式执行：

| 阶段 | 必须先具备 | 入口所执行的顺序 |
| --- | --- | --- |
| `data` | 三个原 CSV、数据依赖 | DuckDB raw 视图 → Parquet → 数据质量 → SQL 质量报告 → EDA → 行序 → adversarial validation → 固定五折 |
| `models` | Parquet、数据库 processed 视图、固定折号 | 常数/逻辑回归 → 核查 → LightGBM 基线及核查 → 候选 init/train/collect → 集成核查 |
| `business` | 两套完整预测/模型元数据及核查、数据质量/漂移来源 | SQL → 策略 → 原 A/B → 严格 A/A → V2；存在冻结清单的阶段改为只读核查 |
| `dashboard` | 完整冻结业务汇总、CSV 和核查来源 | 首次生成聚合 CSV/PBIP；入口已存在时只核查，保留 Desktop 编辑 |

模型阶段只覆盖这次业务冻结依赖。XGBoost/CatBoost、特征筛选和其他竞赛探索有独立脚本及[阶段说明](INDEX.md)，不自动执行。线性基线在模型阶段内部先创建。历史元数据可能包含旧机器绝对路径；现有读取器按当前根目录解析已知文件并核验哈希，不能重写历史来源路径/签名。

约 12.52 GB 原始数据，宽表会额外产生 Parquet、float32 缓存、DuckDB、模型与模拟抽样文件。模型默认 4 个 CPU 线程，使用分批读取和磁盘缓存；缓存不等于全过程恒定低内存。为 raw/processed/cache/实验输出同时预留数十 GB 磁盘，并依据实际机器监测内存。CPU 训练及 100,000 次/场景校准可能运行较久，未提供跨机器耗时承诺。

## 冻结工作区与新版本

当前目录已有五阶段冻结。运行 `reproduce --target business` 时逐段只读检查现有冻结；若核查失败，立即停止。若缺冻结但已有该阶段结果/数据库或下游冻结，也会停止，避免覆盖损坏或不完整的证据。

当前工作区的 `reproduce --target data/models` 被阻止实际执行，因为这些父产物已被业务冻结引用；dry-run 仍显示命令和阻止原因。新数据、参数、条款或实验估计器需要干净工作副本及新版本证据。旧阶段单独脚本可能具备覆盖/恢复选项，不能绕过入口去覆盖冻结父结果；应先保留原工作区，再在新副本完整重建并记录环境差异。

仅核查 V2 可用：

```powershell
.\.venv\Scripts\python.exe scripts/replay_frozen_business.py --stage ab_feasibility_v2 --quick
```

本次增加的入口与文档不改变冻结阶段算法、阈值、样本或决策；新入口会生成普通 Python 缓存，但不写历史结果或审计 JSON。

## Power BI 复现

打开 [LoanRisk.pbip](../dashboard/powerbi/LoanRisk.pbip)，在 Power Query 管理参数中确认 `DataFolder` 指向本机 `dashboard/powerbi/data` 的绝对路径，再点“主页 → 刷新”。聚合数据包不需要完整训练文件或模型。需要联网安装或更新 Desktop 时由用户自行处理。

```powershell
.\.venv\Scripts\python.exe scripts/project.py verify --scope dashboard
```

用户已确认主页面刷新后显示正常；其余五页、PBIX 保存尚未逐项确认。静态核查和 Pillow/Matplotlib 布局示意不能当作新 Desktop 截图或 DAX 执行证据。重建已有报表的直接生成器默认拒绝覆盖，确需重建时先保留 Desktop 编辑，再使用其 `--rebuild`；统一入口默认保留已有报表。详见[使用说明](powerbi_dashboard.md)。

## 本轮交付验收（2026-10-25）

`test --scope business` 的原 73 项通过；随后补充仅源码副本校验，再运行 `test --scope delivery`，13 项通过。两次覆盖合计 13 个交付测试及 61 个既有业务/Power BI 测试，未重训练模型。

`verify --scope business --quick` 五阶段全部通过；检查了现有冻结、行级/SQL、导出与 Bootstrap，A/A/V2 全部重复 Monte Carlo 样本没有本轮重跑。`verify --scope dashboard` 静态核查通过。`status`、`doctor docs/analysis` 和四阶段 dry-run 正常；`doctor full` 如实报告缺少原始 train/test CSV。

文档核查及 19 个可用事实来源的哈希通过，核查结果见 [delivery_verification.json](delivery_verification.json)。只读业务核查后再次核对来源，历史审计保持原哈希。一页 PDF 保留已有视觉验收；架构 PNG 本轮检查布局。真实实验、所有 Desktop 页面验收及从零训练不在本轮完成范围内。

## English execution summary

Use `python scripts/project.py status`, `doctor --profile docs`, and `verify --scope docs` for standard-library source review. `test --scope delivery` checks links, provenance tampering and CLI safety. A source-only copy cannot recompute unavailable private artifacts; the audit reports that scope explicitly.

For execution, create a new virtual environment and install the requirements. Obtain the competition CSVs independently. In a clean work copy, run `reproduce --target data`, then `models`, `business`, and `dashboard`, inspecting each `--dry-run` first. Dependencies are not rebuilt implicitly. Existing frozen business stages are audited read-only; occupied runs without a freeze and data/model rebuilds in a frozen business workspace are blocked. Existing Power BI reports are verified rather than regenerated.

`verify --scope business --quick` validates frozen hashes, rows, SQL, exports, planning and stored calibration rules, including Bootstrap, but skips full repeated Monte Carlo/hash replay. Omit `--quick` for the full read-only replay. Neither command creates real outcomes. Current Parquet/model outputs are available locally; original train/test CSVs are absent. The current Windows lock snapshot differs from historical NumPy training and does not promise bitwise retraining across environments. Refresh PBIP using the local `DataFolder`; only the decision page has user-confirmed runtime acceptance.
