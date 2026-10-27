# 面向公榜 AUC 的优化流程

本轮把用户截图中的官方公榜观察与开发 OOF 分开保存：原 LightGBM 为 0.73212，
三种子调参集成为 0.73453，删除四个漂移列的模型为 0.73501。
删除漂移列在开发 OOF 上稍低、公榜却较高，因此新训练从这个输入方案继续探索。
尚未提交的新文件没有公榜分数，不用 OOF 或预测相关性推算它们的官方成绩。

## 先生成可立即提交的候选

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/build_public_auc_candidates.py
```

脚本复用现有五折 XGBoost 预测，以及已完成的 LightGBM 特征消融和集成。
生成 50:50、70:30 的漂移模型/调参集成概率融合、XGBoost 单模型、
漂移模型/XGBoost 融合、特征可用性对照及两份固定 50:50 排序融合。
每个输出保留 `id,prediction`，
预测类别 1 的还款概率；逐项检查唯一 ID、模板顺序、概率范围、标签方向和 OOF 覆盖。
推荐顺序保存在 `reports/public_auc_optimization/submission_queue.csv`，
推荐依据是新模型的实验角色和已知公榜反馈，不是按开发 OOF 简单取最高值。
队列只包括当前文件哈希尚未取得公榜成绩的候选，已提交的确切版本会自动移出。
相近权重的融合只是备选，不应将当天机会全部用于微调同一组权重。
排序融合先把各模型分数转成经验百分位，再平均；它用于 AUC 排序，
这些百分位不能直接作为业务收益模拟中的还款概率。清单标明融合方式及校准边界。
若新 LightGBM 和 CatBoost 都完成五折，再生成 70:30 概率融合与 50:50 排序融合。
完整 XGBoost 可用时还生成新 LightGBM/XGBoost 的 50:50 融合、
CatBoost/XGBoost 的 50:50 融合，以及 LightGBM/CatBoost/XGBoost 的
固定 50:30:20 概率与排序融合。优先使用深度 8 的完整 CatBoost 候选。
这些都是预先固定的粗粒度权重，没有执行公榜权重搜索。
XGBoost 仍使用原始特征，融合也会引入原模型的漂移敏感性，需要与删漂移列的模型分别提交验证。

文件位于 `submissions/public_auc/`，文件名附加内容 SHA256 前十位。
同一配方产生不同预测时会得到另一文件，不覆盖之前已提交的版本。
`candidate_manifest.json` 保存权重、来源文件校验值、开发指标与输出校验值。
`candidate_metrics.csv` 包括整体、两个已有压力组的 OOF AUC、违约 AP 和预测排序相关性。
`test_rank_correlation_to_public_best` 保留最初删漂移列模型的固定参考；
`test_rank_correlation_to_best_observed_submission` 与当前最高实际公榜成绩对应的文件比较，
该文件的哈希也保存在清单中。相关性描述排序差异，不能据此推算官方 AUC。
`candidate_fold_metrics.csv` 保存每个候选各折及压力组的 AUC，主表同时记录折 AUC 的
均值、最小值和标准差。逐折结果用于识别是否只有个别折带来改善，
不能把已经看过的 OOF 分数作为独立的新测试结果。

## 新训练：按完整五折候选安排预算

```powershell
.\.venv\Scripts\python.exe scripts/run_public_auc_optimization.py --stage init
.\.venv\Scripts\python.exe -u scripts/run_public_auc_optimization.py --stage train --time-budget-minutes 90 --parallel
```

默认单进程、10 个 CPU 线程；`--parallel` 使用两个训练进程、每进程 5 线程，
总预算仍为 10 个逻辑 CPU。GPU 训练最多同时运行一个任务。
并发模式启动前至少需要 6GiB 可用内存，内存规划不能替代进程实际内存监测。
主进程只发送命令和文件路径，各子进程按批读取只读缓存，退出时释放原生库和 GPU 资源。

训练任务按以下次序安排：

1. `lgbm_drift_tuned`，种子 42、完整五折：删除 `feature_492/903/586/1039`，
   使用上一轮每折选出的参数和树数，在新的特征子集上完整重训。
2. `catboost_drift_gpu`，种子 42、完整五折：同样删除四列，使用数值特征的
   Plain boosting、深度 6、63 个分箱边界、原生 NaN 处理。
   内部拟合/早停为 85%/15%，内部 AUC 选择最多 1,200 棵树，再在完整拟合四折重训。
3. `lgbm_drift_tuned`，种子 2026、3407：形成同折多种子概率平均和五折测试平均。
4. `lgbm_drift_available`：四个漂移列 + 测试全空列 + 当前拟合四折缺失率至少 90% 的列。
5. `lgbm_drift_bins255`：四个漂移列 + 255 分箱 + 50,000 行拟合分箱样本。

优先完成每个候选的五折，避免把部分折成绩误当全量 OOF。
时间允许时继续后续变体，不以某个外层分数上涨为由临时增加搜索。
这里的 LightGBM 使用历史选好的参数和树数，属于固定配方的特征/分箱实验，
不能称为对新特征空间重新执行过完整的超参数搜索。

主进程执行墙钟时间上限，到期只终止它自己启动的未完成子进程。
已完成折不丢失；再次运行 train 自动核对校验值并续跑。
Windows 下会同时停止虚拟环境启动器及其 Python 子进程，避免到期后留下训练进程。
只有完成五折的单种子模型，或至少两个完整种子的集成，才会导出新提交。
每出现一个新的完整候选便刷新候选表；最终还生成与原公榜最好模型的固定概率融合。

## GPU 与输入数据核查

本机检测到 RTX 3060 Laptop GPU、6GB 显存。
CatBoost 的小数据量化 Pool GPU 试跑通过；XGBoost 的当前安装请求 GPU 后会回退 CPU，
因此本轮复用已完成的 CPU XGBoost，不把它记作 GPU 加速。
CatBoost GPU 不支持本轮二分类的 `rsm`，也不支持 Python 早停回调，
所以使用原生 `od_type=Iter`、`od_wait=60`；这些配置与 CPU 库的早停语义不同。
实际后端、参数、树数和耗时均记录在每折 checkpoint 中，GPU 不保证逐位确定性。

### 二进制 GPU 输入实验

本机正式试跑中，TSV 导出成为主要耗时，预计无法在本轮预算内完成五折。
该任务已主动停止，决定记录在 `gpu_preparation_budget_decision.json`。
原流程的 GPU 调度已停用；硬件探测通过保存在 `gpu_probe.hardware_available`，
以后续跑 CPU 候选不会再次进入耗时的 TSV 准备。
另提供独立实验，避免改写既有训练配置和模型：

```powershell
.\.venv\Scripts\python.exe -u scripts/run_catboost_fast_gpu.py --stage train --time-budget-minutes 40
.\.venv\Scripts\python.exe scripts/run_catboost_fast_gpu.py --stage verify
```

该流程使用 float32 列优先 mmap 准备数值，再仅在拟合数据上量化，保存并重载二进制 Pool，
释放原始数据引用；验证数据复用拟合分箱边界。小数据测试核对了与 TSV 的边界及 CPU 模型
预测完全一致，并检查了真实零、缺失值、标签方向和验证常量特征。
全量运行的实际耗时与内存须以任务日志为准。
另有真实 GPU 小数据五折试跑，完成模型重载并重现预测；证据位于
`reports/public_auc_optimization/fast_gpu_smoke_test.json`，该证据不代表全量数据已完成训练。
GPU worker 默认限制到 10 个逻辑 CPU；量化接口没有线程数参数，因此还设置进程 CPU 亲和性。
该命令应在原训练进程结束后运行，以维持本机总资源预算。
若同时续训 CPU 种子，在两个终端分别运行以下命令，总预算为 5+5 个逻辑 CPU：

```powershell
.\.venv\Scripts\python.exe scripts/run_public_auc_cpu_refinements.py --family seeds --time-budget-minutes 45
.\.venv\Scripts\python.exe scripts/run_catboost_fast_gpu.py --stage train --threads 5 --time-budget-minutes 45
```

CPU 续训默认单进程、5 线程；独立运行时可加 `--parallel` 使用两个进程各 5 线程，
这时应独占训练资源。`--family availability` 和 `--family bins` 可续训另外两个特征实验。
CPU 控制器仅保存折 checkpoint，两个控制器结束后运行原脚本 `--stage collect` 汇总，
避免并发改写同一集成预测文件。

### 固定容量 Challenger

GPU 配方达到 1,200 轮上限后，增加独立的深度 8、最多 2,000 轮、L2=8 的固定配置：

```powershell
.\.venv\Scripts\python.exe scripts/run_catboost_gpu_capacity.py --stage train --threads 2 --time-budget-minutes 35
.\.venv\Scripts\python.exe scripts/run_public_auc_cpu_refinements.py --family availability --parallel --threads 4 --time-budget-minutes 30
```

两个 CPU 训练进程各 4 线程，GPU 工作进程最多使用 2 个逻辑 CPU，总预算为 10。
容量实验复用同一拟合行、标签、特征与分箱的已核查二进制 Pool；训练参数与输出隔离，
保持原训练引擎源码和已完成模型。GPU 内存比例降低为 60%，为桌面显示保留空间。
这是固定容量对照，没有执行完整超参数搜索。
在隔离的小数据 GPU 试跑中核查了分箱缓存未改写、模型后端与重载预测，
证据为 `reports/public_auc_optimization/capacity_gpu_smoke_test.json`。
正式结果保存在 `artifacts/catboost_gpu_capacity/` 与 `reports/catboost_gpu_capacity/`，
只有完成五折才会导出 `catboost_gpu_capacity_s42`。
核查命令为 `scripts/run_catboost_gpu_capacity.py --stage verify`。
只在五折完成后导出 `catboost_fast_gpu_s42`，完整输出自动接入融合脚本。
实验文件在 `artifacts/catboost_fast_gpu/` 和 `reports/catboost_fast_gpu/`。

### 公榜 0.73949 之后的多种子对照

2026-10-25 用户截图显示三模型融合取得公榜 0.73949、第二名，第一名为 0.74111。
此次反馈与确切提交文件哈希一起记录；截图显示最终可选最多 3 个文件，
与初始大纲中的 2 个不同，最终选择应以 Kaggle 当前页面和规则为准。

继续使用相同深度 8 配方，增加种子 2026、3407，每个种子完整五折：

```powershell
.\.venv\Scripts\python.exe -u scripts/run_catboost_gpu_multiseed.py --stage train --seeds 2026 3407 --threads 2 --time-budget-minutes 95
.\.venv\Scripts\python.exe -u scripts/run_public_auc_cpu_refinements.py --family bins --threads 8 --time-budget-minutes 95
```

两条命令可同时运行，总 CPU 预算为 2+8。本轮可用内存约 6GB，
因此 CPU 分箱对照使用单个工作进程；GPU 一次只训练一折。
这是本机的资源选择，并不表示两个 4 线程 CPU 工作进程已通过本轮内存预检。
两个新增 GPU 种子共用一个总时间上限，到期停止其拥有的工作进程并保留完成折。
种子改变训练随机性，内部切分、拟合行、特征及训练量化边界保持一致。

新模型保存在 `artifacts/catboost_gpu_multiseed/seed_<seed>/jobs/`，
配置和进度在 `reports/catboost_gpu_multiseed/`。至少两个完整种子时导出
`catboost_gpu_capacity_ensemble`，可以包含原始种子 42。
融合流程优先使用这个完整集成，沿用已验证的 50:30:20 配方，
同时保留 20:40:40 的粗粒度对照；新内容得到新文件哈希，原 0.73949 版本保留。

训练控制器结束后再汇总 CPU 对照，以免并发写入相同候选文件：

```powershell
.\.venv\Scripts\python.exe scripts/run_public_auc_optimization.py --stage collect
.\.venv\Scripts\python.exe scripts/run_catboost_gpu_multiseed.py --stage verify --seeds 2026 3407
.\.venv\Scripts\python.exe scripts/verify_public_auc_optimization.py
.\.venv\Scripts\python.exe scripts/verify_public_auc_candidates.py
```

GPU 小数据多种子核查包括完整五折平均、种子身份、模型重载和缓存不变，
证据为 `reports/public_auc_optimization/multiseed_gpu_smoke_test.json`。
该试跑使用 500 行和 8 轮上限，不能代替正式数据上的模型成绩。

用户随后确认 CatBoost/XGBoost 50:50 的实际公榜为 **0.74009**。
后续队列优先使用这个模型组合的新增完整种子和深度对照。
深度 9 配方独立保存，沿用训练量化缓存，最多 2,200 轮、L2=12；GPU 任一时刻只运行一项训练。
在 CPU 分箱任务结束后，可以同时安排以下两项，仍保持 2+8 CPU 预算：

```powershell
.\.venv\Scripts\python.exe scripts/run_catboost_gpu_depth9.py --stage train --threads 2 --time-budget-minutes 45
.\.venv\Scripts\python.exe scripts/run_xgboost_drift_fast.py --stage train --threads 8 --time-budget-minutes 55
```

新 XGBoost 删除四个漂移列，使用固定 600 轮、0.075 学习率、深度 6、叶子 L2=8。
只训练一次完整拟合四折，轮数由固定预算预先指定；外层验证不参与早停。
这是整套固定配方对照，不能把变化单独归因于列删除或某一项参数。
控制器结束后通过 `--stage collect` 导出完整 OOF 与测试分数，`--stage verify` 重载模型核对预测。
新 XGBoost 的训练控制器也推迟共享预测导出，避免其他模型读取正在写入的 Parquet。

项目迁移后时间戳被截断。初始化会逐批把旧 float32 缓存的**全部数值**与当前
Parquet 转为 float32 后的数值比较，再记录文件内容校验值，避免仅凭行列数或文件大小复用。
缓存和旧实验模型保持原样。所有模型仍排除 id、label、fold 和压力组标记，
所有特征仍为匿名数值，不虚构业务含义。

## 进度、反馈和产物

```powershell
Get-Content reports/public_auc_optimization/training_progress.json
.\.venv\Scripts\python.exe scripts/run_public_auc_optimization.py --stage collect
.\.venv\Scripts\python.exe scripts/verify_public_auc_optimization.py
.\.venv\Scripts\python.exe scripts/verify_public_auc_candidates.py
```

训练配置：`reports/public_auc_optimization/training_config.json`。
完整五折模型：`artifacts/public_auc_optimization/completed/`。
每折模型与预测：`artifacts/public_auc_optimization/jobs/`。
逐任务日志：`logs/public_auc_optimization/<会话编号>/`。
重载核查：`reports/public_auc_optimization/model_reload_verification.json`；核查已完成模型的
文件校验值、特征顺序和树数，重算完整 OOF 指标，并抽样重现 OOF/测试预测。
提交文件核查独立重读所有当前候选 CSV，按来源和权重重新计算完整测试分数及 OOF AUC，
核对文件哈希、来源哈希、所有 ID 和模板行序。结果保存到
`reports/public_auc_optimization/candidate_file_verification.json`。

用户提交后用实际返回分数登记，例如：

```powershell
.\.venv\Scripts\python.exe scripts/build_public_auc_candidates.py --record-submission public_auc/实际文件名.csv --public-score 实际返回分数
```

真实数值需替换命令中的占位内容。记录同时保存提交文件的 SHA256，
避免同名文件变化后把旧分数归给新预测。初始三个分数来源为用户截图，
后续分数由用户录入；脚本没有自动上传任何提交，也没有获得测试标签。
公榜只覆盖部分测试数据，迭代结果仍可能受到榜单抽样和多次选择影响。
最终选择两份提交时应结合不同方案的表现与预测差异，保留验证边界。
