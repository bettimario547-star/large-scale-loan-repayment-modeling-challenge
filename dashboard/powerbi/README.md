# 贷款风险决策 Power BI 项目

入口：[LoanRisk.pbip](LoanRisk.pbip)。打开项目后点击 **刷新**，载入 `data/` 中的 23 张小型聚合表。默认使用本项目的绝对数据路径；移动项目后，先在 Power Query 的“管理参数”中修改 `DataFolder`，再刷新。

本交付是六页可编辑的 PBIP/PBIR 报表及本地导入语义模型，尚未确认 Desktop 保存 PBIX。无需登录或发布到云端。文件结构、官方 schema、源数据、空值、字段引用与参考数学检查已完成。**2026-10-25 用户确认“主页 → 刷新”后可以正常显示，主页面的刷新和度量显示已验收；其余五页仍未逐页确认。**

已修复缺少 `definition/version.json`、`KS` 列/度量重名，以及缺少增强数据源元数据标记的问题。当前模型明确声明 `defaultPowerBIDataSourceVersion: powerBI_V3`。如果仍打开着修复前的项目，请先关闭（不要保存报错后的挂起更改），重新打开入口再刷新，避免旧缓存覆盖文件修复。尚未载入数据时，决定卡片显示“等待数据刷新”。

| 页面 | 展示内容 |
| --- | --- |
| 01 决策总览 | 固定收益区间、样本完成情况、八项晋级条件；保留 Champion |
| 02 模型历史表现 | 复用 OOF 的 AUC/AP/KS，历史风险分箱、无标签测试构成、分数 PSI |
| 03 审批策略与经济假设 | 冻结阈值、不同经济情景、全评价收益/风险曲线；测试收益为空 |
| 04 A/A 校验与停止规则 | 83 项校准结果，六项未认证压力检查，未经调整提前停止的误报 |
| 05 A/B 可行性与周期假设 | 原方案与 V2，流量/合格比例/分歧暴露控件，成熟期与宽限 |
| 06 收益区间、晋级与真实数据缺口 | 固定 V2 区间、历史与合成模拟成功频率、真实数据需求 |

默认流量是每日 5,000 申请、共同合格占比 20%、分歧全部入组。切片器未选唯一值时，度量使用此默认值；切片器一旦选值，只改变周期假设。经济情景默认基准，模型默认展示两者。收益证据、5 元业务门槛和晋级决定不随这些控件改变。

**证据边界：**历史标签/OOF、现金流及流量假设、真实业务待验证项分别标注。当前收益是历史离线模拟，不能称作真实 A/B 因果收益。测试坏账、测试收益、真实现金流和 DPD 均未观测；未知项保持空值。

完整说明、指标口径与 Desktop 验收步骤：[docs/powerbi_dashboard.md](../../docs/powerbi_dashboard.md)。

## 复现与保护

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_powerbi_dashboard.py
.\.venv\Scripts\python.exe -m unittest tests.test_powerbi_dashboard -v
.\.venv\Scripts\python.exe scripts\render_powerbi_preview.py
```

`scripts/build_powerbi_dashboard.py` 默认拒绝覆盖已有项目。仅在先备份 Desktop 编辑后，使用 `--rebuild` 重新生成报表、模型和聚合数据。生成器读取冻结审计材料，不训练模型、不改动原实验冻结文件。

`layout_previews/` 是 Pillow/Matplotlib 生成的布局及默认数据示意，**不是 Power BI 截图，也不是 DAX 执行证据**。`artifact_verification.json` 和 `source_manifest.json` 记录具体检查范围及未验证项。
