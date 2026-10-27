# 严格 A/A 离线校验

本阶段专门验证实验管线，保留 Champion，不优化 A/B，不晋级 Challenger。
两组执行同一 Champion 预测、阈值、审批决定和假设现金流；原 A/B 共同代理范围/风险层保持冻结。
原 `score_b` 是入组协变量，实际 B 组处理分数单独存为 `treatment_score_b=score_a`。
这保证校验同一目标范围，避免为了 A/A 改动人群导致结果不能用于当前管线。

## 运行

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_aa_validation tests.test_ab_test_design tests.test_approval_policy_simulation tests.test_sql_business_analysis -v
.\.venv\Scripts\python.exe scripts/run_aa_validation.py
.\.venv\Scripts\python.exe scripts/verify_aa_validation.py
```

命令也可在 cmd.exe 中执行。
配置在 `configs/aa_validation.json`。原 A/B 协议、数据、模型和冻结文件均只读。
再次运行深度核查并复用同一快照；改变校验代码/配置需新 `--run-name aa_validation_v2`。
默认包含 64 次实际哈希分流和每场景 100,000 次统计模拟，运行会持续输出进度。

## 严格标准

- 单次完整管线：逐申请验证同一政策、冻结准入、唯一数据集/ID、稳定分组、无标签泄漏、ITT 分母、拒绝成本和未知逾期 NULL。
- 真实哈希：64 个预声明盐值，完整核验每次分流；频次精度有限，仅作实现抽查。
- 必检统计场景：当前完整有限总体 Bernoulli 分流；同一 Champion 经验分布的当前规模独立组；原规划规模独立组。
- 端点：收益 Welch、审批率零差 Wald、ITT 违约发生率零差 Wald，检验水平 1%/5%/10%。
- 验收：所有 Clopper–Pearson 同时区间完整落入预声明 FPR 容许范围，且无无效重复。家族置信水平至少 95%。
- 95% 区间覆盖率是相应检验接受率的补数，不能当作额外独立证据。
- SRM：整体/风险层快照按 Bonferroni；有限总体重复校准整体 SRM 误报率上界。
- 故障注入：错误收益、分流、标签决定分组/准入、重复 ID、伪造逾期、错误分母和不合规提前分析，必须被相应检查发现。

一次 p>0.05 不能证明管线可靠。零效应也会按预定概率显著，不要求全部盐值都不显著，禁止按结果换种子。
两组同策略导致配对收益差逐行等于零，只能作为身份检查；独立分组 SE 仍为正，不能使用配对零方差规划样本量。
原非劣效假设在 A/A 零差下属于备择内部，其通过率不等于边界假阳性率；此边界校准尚未完成。
Bootstrap 只做一次独立组稳健性演练，不声称已经完成 Bootstrap 全场景覆盖率认证。

## 本轮结果与解释

当前合格评价总体为 210,856 个申请。三个必检场景的 27 项端点/alpha 检查均通过预声明认证：
5% 水平的收益检验误报率分别为 5.061%（当前有限总体）、5.039%（当前规模独立组）、
5.118%（原规划规模独立组）。分流比例检查、17 类负向控制也通过。
本轮 14 项 A/A 测试与原 SQL、审批策略和 A/B 模块的 24 项回归测试全部通过，共 38 项。

总样本 10,000、B 占 10% 的压力场景未认证，收益检验在 5% 水平的误报率为 5.536%，
同时置信区间为 [5.296%, 5.783%]，未完整落入 [4.5%, 5.5%] 验收范围。
同一场景的 ITT 违约发生率检验也未认证；其余三个压力场景通过。不能把本轮结论推广为所有分配比例和样本规模都可靠。

一次 A/A 快照的收益差为 -7.13 元，95% 区间 [-20.27, 6.02]，p=0.2881。
同一次快照的 ITT 违约发生率非劣效诊断未通过：单侧上界为 0.2255 个百分点，超过 0.20 个百分点界值。
该快照的样本量低于原护栏规划的 221,670 个申请；零差处于非劣效备择内部，同策略也可能因抽样波动未通过非劣效诊断。
这与零差检验的误报率校准属于不同问题，既不能据此否定已通过的零差校准，也不能声称护栏已获得业务安全认证。
完整诊断保留在 `reports/aa_validation/summary.json`。

五次未经调整的累计查看，使双侧误报率由固定终点的 4.934% 升至 14.20%。
因此保留固定样本、结果成熟、质量通过后的最终分析门槛。Champion 继续保留，后续 A/B 优化及新模拟尚未执行。

## 压力场景和停止规则

预声明 10,000/50,000 申请、B 分配 10%/50% 的压力场景，全部报告。
压力场景未认证可代表近似检验失准或精度不足；不会通过移除场景来掩盖失败。
必检场景的认证只适用于对应范围；未来改估计器、人群、比例、样本规模或停止规则，应重新做匹配的 A/A。

通过累计样本流模拟未经调整的 5 次查看，分别显示双侧误报和正向“成功”误报：单次名义水平为 5% 与 2.5%。
该可选停止演示有自己的 100,000 申请固定终点，与原 A/B 的约 372 万样本量计划分开；不代表原正式试验已达到样本要求。
这只是可选停止错误的演示；真实协议仍只允许固定样本达到、全部结果成熟、质量通过后的最终收益分析。
安全暂停不等于提前成功。历史数据没有成熟日期，无法替代真实成熟校验。

## 数据与交付

数据库 `data/interim/aa_validation.duckdb`，报告 `reports/aa_validation/aa_validation_report.md`。

| 表 | 用途 |
|---|---|
| experiment_population / experiment_assignments | A/A 克隆总体/离线实际哈希快照，复用原实验执行 SQL |
| experiment_metrics / eligibility_summary | ITT 指标和原范围覆盖；两组处理为 Champion |
| assignment_quality / risk_score_concentration | 整体/分层分流与获批风险分数结构 |
| aa_covariate_balance | 分组前分数的 SMD |
| aa_null_cells | 同 Champion 现金流/审批/违约事件单元与人数 |
| aa_hash_replays | 64 次真实哈希分流的组人数、差值、p、区间和 SRM |
| aa_calibration_results | 场景×端点×alpha 的假阳性率、同时区间、覆盖率和认证 |
| aa_pvalue_histogram | p 值十分位；仅诊断，不单凭图形认证 |
| aa_srm_calibration | 理想 Bernoulli 分流的 SRM 误报与精确区间 |
| aa_snapshot_bootstrap | 独立组快照 Bootstrap，配对零差不进入统计推断 |
| aa_optional_stopping | 累计样本的固定终点与未经调整提前查看误报 |
| aa_negative_controls | 隔离故障注入是否被发现 |

13 份 CSV 在报告的 tables/；两个 PNG 在 figures/。
`artifacts/aa_validation/` 保存分流快照、重复样本、协议、治理决定和 `aa_freeze_manifest.json`。
`model_governance_decision.json` 记录保留 Champion、暂无 Challenger 晋级，未对外部系统采取生产动作。
哈希锁定父阶段数据、执行代码、验收规则、数据库、报告和导出；独立 verifier 重算统计模拟和所有哈希抽查。

有限总体单元二项分流与理想逐行 Bernoulli 分流数学等价，不是重复执行真实 SHA256。
有放回独立组模拟不增加真实数据，固定历史分布上的校准不能覆盖未来漂移、重复客户相关性、时间结构和原获批样本选择偏差。
借款人标识、真实逾期、成熟现金流和业务安全范围仍需前瞻数据；离线 A/A 通过不等于线上 A/A 或生产就绪。
