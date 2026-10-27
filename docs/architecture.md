# 项目架构 / Project architecture

[交付导航](INDEX.md) · [模型卡](model_card.md) · [实验卡](experiment_card.md) · [复现指南](reproducibility.md)

项目是一条本地批处理链路：从匿名 Kaggle 数据生成固定 OOF，再将预测冻结为 SQL/策略/实验输入。后续阶段复用冻结结果，不再训练模型。原 A/B → 严格 A/A → V2 保留版本演进；新估计器必须有匹配校准，旧 A/A 不能直接认证 V2。

![架构图](figures/architecture.svg)

[PNG 版本](figures/architecture.png) · [可重绘图源](../scripts/render_project_architecture.py)

## 可编辑 Mermaid 源码

```mermaid
flowchart TD
    D["Kaggle CSV · anonymous numeric features<br/>label 1=repayment · id only for joins"]
    Q["DuckDB + Parquet<br/>SQL quality / EDA / adversarial validation"]
    F["Fixed shuffled stratified 5 folds<br/>inner early stopping · row order not verified as time"]
    M["Champion: LightGBM 5 folds<br/>Challenger: 5 folds x 3 seeds<br/>reused development OOF · uncalibrated"]
    P["SHA-256 frozen predictions<br/>(dataset, id, model_version)"]
    S["SQL risk bands / AUC / AP / KS / score drift"]
    A["Approval simulation<br/>development/evaluation split<br/>frozen thresholds + cash-flow assumptions"]
    B["Original A/B design and replay"]
    AA["Strict same-Champion A/A<br/>SRM / fault injection / peeking / calibration"]
    V["V2 disagreement-only randomization<br/>stratified independent arms<br/>standardize to full-eligible ITT"]
    C["V2 matched A/A + synthetic business/NI boundaries<br/>lock design before evaluation replay"]
    R["V2 feasibility + fixed-endpoint simulation<br/>profit CI / guardrails / sample + maturity gates"]
    O["Retain Champion<br/>model / experiment cards · one-page memo<br/>6-page Power BI · read-only reproduction entry"]
    L["Future live data<br/>borrower_id · safety · execution<br/>actual cash flows · mature DPD"]
    T["Future: shadow -> independent pilot -> locked main trial<br/>all promotion gates · NOT IMPLEMENTED"]
    D --> Q --> F --> M --> P --> S --> A --> B --> AA --> V --> C --> R --> O
    L -.-> T
    O -. "new data and new protocol required" .-> T
    classDef historical fill:#eef4fb,stroke:#35668f,color:#132f49;
    classDef simulation fill:#fff6e7,stroke:#ae7629,color:#553817;
    classDef decision fill:#eaf5ef,stroke:#397454,color:#183f2b;
    classDef future fill:#f5f5f5,stroke:#7a838c,stroke-dasharray:5 4,color:#47525d;
    class D,Q,F,M,P,S historical;
    class A,B,AA,V,C,R simulation;
    class O decision;
    class L,T future;
```

## 依赖、分母和冻结边界

| 层 | 主实现 / 输入 | 输出及边界 |
| --- | --- | --- |
| 数据与验证 | `scripts/init_duckdb.py` 至 `build_validation_strategy.py` | Parquet、质量/漂移报告、固定折号；无可信真实时间列 |
| 模型 | 线性 → LightGBM → 有限候选/多种子 | 训练 OOF、无标签测试预测、训练签名；开发 OOF 被复用 |
| SQL | 两套模型预测及模型文件来源 | SHA-256 冻结、风险分箱和模型指标；未知测试标签/收益为 NULL |
| 策略 | 冻结预测、`approval_policy_simulation.json` | 开发 413,398 / 评价 275,600；阈值、固定成本/收益、政策交接 |
| 原实验 / A/A | 策略交接、相同 Champion 执行 | 原方案规划、误报/停止规则、压力边界；保持原记录 |
| V2 | 原共同合格人群、预分析锁、分歧方向与风险层 | 实际随机分歧 12,614、完整合格评价 210,856；按完整 ITT 权重推断；规划 17,141 |
| 展示与交付 | 冻结汇总/审计、23 张聚合表 | Power BI、事实快照和卡片；收益是模拟，真实晋级未完成 |

图中的数据流不代表已部署实时服务。当前没有客户级身份、真实成熟现金流或生产审批系统；未来链路以虚线表示。

冻结文件同时约束父输入、配置/代码及结果。`project.py reproduce --target business` 对已有冻结阶段执行只读核查；数据/模型重建会改动业务父来源，所以在有业务冻结的当前工作副本中阻止执行。新数据/模型/政策版本应在干净工作副本生成新证据，不能改写旧冻结清单来通过检查。

## English reading note

Blue blocks are historical data/model analysis. Amber blocks are offline experiments with assumed cash flows. Green is the retained-Champion decision and portfolio delivery. Dashed gray blocks are proposed live implementation, which has not happened. V2 randomizes disagreements only but estimates profit at the **full common-eligible ITT denominator**, requiring agreement policies to have identical terms and cash flows.
