"""Evaluate frozen approval policies under explicit, editable offline economics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

try:
    from . import policy_simulation_common as policy
    from .verify_approval_policy_simulation import verify
    from .verify_sql_business_analysis import verify as verify_parent
except ImportError:
    import policy_simulation_common as policy
    from verify_approval_policy_simulation import verify
    from verify_sql_business_analysis import verify as verify_parent


def build_database(output, config, candidates):
    output.database.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.database.with_name(f".{output.run_name}_{time.time_ns()}.duckdb")
    scratch = output.artifacts / "duckdb_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with duckdb.connect(str(temporary)) as connection:
            for setting, value in (("threads", config["threads"]), ("memory_limit", config["memory_limit"]), ("temp_directory", str(scratch))):
                connection.execute(f"SET {setting} = ?", [value]).fetchall()
            connection.execute("BEGIN TRANSACTION").fetchall()
            selected = policy.execute_sql(connection, output, config, candidates)
            connection.execute("COMMIT").fetchall()
            connection.execute("CHECKPOINT").fetchall()
        temporary.replace(output.database)
        return selected
    finally:
        if temporary.exists(): temporary.unlink()


def export_results(output):
    tables = output.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(output.database), read_only=True) as connection:
        for table in policy.EXPORT_TABLES:
            connection.execute(f"COPY {table} TO ? (FORMAT CSV, HEADER)", [str(tables / f"{table}.csv")]).fetchall()
        connection.execute("COPY selected_policy_application_outcomes TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                           [str(output.artifacts / "selected_policy_application_outcomes.parquet")]).fetchall()


def render_figures(output, config, predictions, split):
    cache = output.artifacts / "matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    destination = output.reports / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(output.database), read_only=True) as connection:
        curves = connection.execute("SELECT * FROM approval_economic_results WHERE scenario_id='base' AND policy_cohort='policy_evaluation' ORDER BY target_approval_rate").fetchdf()
        selected = connection.execute("SELECT * FROM selected_policy_results WHERE policy_cohort='policy_evaluation'").fetchdf()
    colors = {"champion": "#3266a8", "challenger": "#d47b23"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for role, color in colors.items():
        rows = curves.loc[curves.model_role.eq(role)]
        axes[0].plot(rows.approval_rate * 100, rows.observed_default_rate_approved * 100, marker="o", color=color, label=role)
        axes[1].plot(rows.observed_default_rate_approved * 100, rows.historical_profit_per_application, marker="o", color=color, label=f"{role}: label-based simulation")
        axes[1].plot(rows.observed_default_rate_approved * 100, rows.raw_score_implied_profit_per_application, linestyle="--", color=color, label=f"{role}: raw-score diagnostic")
        chosen = selected.loc[selected.scenario_id.eq("base") & selected.model_role.eq(role) & selected.prospective_arm.isin(["A", "B"])].iloc[0]
        axes[0].scatter(chosen.approval_rate * 100, chosen.observed_default_rate_approved * 100, s=180, marker="*", color=color, edgecolor="black", zorder=5)
        axes[1].scatter(chosen.observed_default_rate_approved * 100, chosen.historical_profit_per_application, s=180, marker="*", color=color, edgecolor="black", zorder=5)
    axes[0].axhline(config["constraints"]["max_development_default_rate"] * 100, linestyle=":", color="#b44", label="Development risk limit")
    axes[1].axvline(config["constraints"]["max_development_default_rate"] * 100, linestyle=":", color="#b44")
    axes[1].axhline(0, color="gray", linewidth=.8)
    axes[0].set(title="Frozen thresholds on policy-evaluation OOF", xlabel="Approval rate (%)", ylabel="Observed default rate among approved (%)")
    axes[1].set(title="Base assumptions: simulated profit and risk", xlabel="Observed default rate among approved (%)", ylabel="Simulated net profit per application (assumed CNY)")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure.savefig(destination / "approval_risk_profit.png", dpi=160); plt.close(figure)

    comparison = selected.loc[selected.prospective_arm.isin(["A", "B"])].pivot(index="scenario_id", columns="prospective_arm", values="historical_profit_per_application")
    comparison = comparison.loc[[s["scenario_id"] for s in config["sensitivity_scenarios"]]]
    labels = {"base": "Base", "lower_interest": "Interest 8%", "full_principal_loss": "LGD 100%", "higher_funding": "Funding 6%", "smaller_principal": "Principal 5k", "larger_principal": "Principal 20k"}
    positions = np.arange(len(comparison))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].bar(positions - .18, comparison.A, width=.36, color=colors["champion"], label="A: frozen Champion policy")
    axes[0].bar(positions + .18, comparison.B, width=.36, color=colors["challenger"], label="B: frozen Challenger policy")
    axes[0].set_xticks(positions, [labels.get(s, s) for s in comparison.index], rotation=20, ha="right")
    axes[0].set(title="Same policies under different assumptions", ylabel="Simulated profit per application (assumed CNY)")
    delta = comparison.B - comparison.A
    axes[1].barh(positions, delta, color=colors["challenger"])
    axes[1].set_yticks(positions, [labels.get(s, s) for s in comparison.index])
    axes[1].set(title="B minus A; no scenario-specific re-selection", xlabel="Simulated profit difference per application")
    for axis in axes: axis.axvline(0, color="gray", linewidth=.8) if axis is axes[1] else axis.axhline(0, color="gray", linewidth=.8); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.savefig(destination / "economic_sensitivity.png", dpi=160); plt.close(figure)

    evaluation_ids = set(split.loc[split.policy_cohort.eq("policy_evaluation"), "id"])
    evaluation = predictions.loc[predictions.dataset.eq("train") & predictions.id.isin(evaluation_ids)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    gains, lift = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for role, color in colors.items():
        source = evaluation.loc[evaluation.model_role.eq(role)]
        y, score = 1 - source.label.to_numpy(dtype=np.int8), source.p_default.to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        indices = np.linspace(0, len(recall) - 1, min(800, len(recall))).astype(int)
        axes[0].plot(fpr, tpr, color=color, label=f"{role}: AUC {roc_auc_score(y, score):.4f}")
        axes[1].step(recall[indices], precision[indices], where="post", color=color, label=f"{role}: AP {average_precision_score(y, score):.4f}")
        grouped = pd.DataFrame({"score": score, "default": y}).groupby("score").agg(rows=("default", "size"), defaults=("default", "sum")).sort_index(ascending=False)
        fraction = grouped.rows.cumsum().to_numpy() / len(y)
        capture = grouped.defaults.cumsum().to_numpy() / y.sum()
        idx = np.linspace(0, len(fraction) - 1, min(800, len(fraction))).astype(int)
        lift[0].plot(np.r_[0, fraction[idx]] * 100, np.r_[0, capture[idx]] * 100, color=color, label=role)
        lift[1].plot(fraction[idx] * 100, capture[idx] / fraction[idx], color=color, label=role)
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[1].axhline(np.mean(y), color="gray", linestyle=":", label="Default prevalence")
    axes[0].set(title="Policy-evaluation OOF: default ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Policy-evaluation OOF: default precision-recall", xlabel="Recall", ylabel="Precision")
    for axis in axes: axis.legend(fontsize=8); axis.grid(alpha=.2)
    figure.savefig(destination / "roc_pr.png", dpi=160); plt.close(figure)
    lift[0].plot([0, 100], [0, 100], color="gray", linestyle=":", label="Random ordering")
    lift[1].axhline(1, color="gray", linestyle=":")
    lift[0].set(title="Cumulative defaults captured, high-risk first", xlabel="Applications prioritized (%)", ylabel="Defaults captured (%)")
    lift[1].set(title="Cumulative default lift, high-risk first", xlabel="Applications prioritized (%)", ylabel="Lift relative to default prevalence")
    for axis in lift: axis.legend(fontsize=8); axis.grid(alpha=.2)
    gains.savefig(destination / "lift_gains.png", dpi=160); plt.close(gains)


def write_report(output, config, selected, split, audit):
    with duckdb.connect(str(output.database), read_only=True) as connection:
        comparison = connection.execute("SELECT prospective_arm, policy_cohort, threshold, approval_rate, observed_default_rate_approved, historical_profit_per_application, raw_score_implied_profit_per_application FROM selected_policy_results WHERE scenario_id='base' ORDER BY prospective_arm, policy_cohort").fetchdf()
        sensitivity = connection.execute("SELECT scenario_id, prospective_arm, historical_profit_per_application FROM selected_policy_results WHERE policy_cohort='policy_evaluation' AND prospective_arm IN ('A','B') ORDER BY scenario_id, prospective_arm").fetchdf()
        paired = connection.execute("SELECT target_approval_rate, exactly_equal_approved_count, delta_approval_rate, delta_default_rate, delta_profit_per_application FROM matched_approval_comparison WHERE scenario_id='base' AND policy_cohort='policy_evaluation' ORDER BY target_approval_rate").fetchdf()
        test_applications = int(connection.execute("SELECT total_applications FROM selected_policy_results WHERE scenario_id='base' AND policy_cohort='unlabeled_test' AND prospective_arm='A'").fetchone()[0])
    base = policy.economics_frame(config).query("scenario_id == 'base'").iloc[0]
    a = comparison.query("prospective_arm == 'A' and policy_cohort == 'policy_evaluation'").iloc[0]
    b = comparison.query("prospective_arm == 'B' and policy_cohort == 'policy_evaluation'").iloc[0]
    test_a = comparison.query("prospective_arm == 'A' and policy_cohort == 'unlabeled_test'").iloc[0]
    test_b = comparison.query("prospective_arm == 'B' and policy_cohort == 'unlabeled_test'").iloc[0]
    delta = float(b.historical_profit_per_application - a.historical_profit_per_application)
    counts = split.policy_cohort.value_counts().to_dict()
    feasible = {arm: bool(row.observed_default_rate_approved <= config['constraints']['max_development_default_rate'] and row.approval_rate >= config['constraints']['min_development_approval_rate']) for arm, row in (("A", a), ("B", b))}
    definitions = selected.loc[selected.prospective_arm.isin(["A", "B"]), ["prospective_arm", "model_role", "model_version", "target_approval_rate", "threshold", "decision_rule"]]
    economics = pd.DataFrame([{"assumption": field, "value": config["economics"][field]} for field in config["economics"]])
    evaluation_display = pd.DataFrame([
        {"策略": arm, "评价审批率": row.approval_rate, "获批历史违约率": row.observed_default_rate_approved,
         "每申请模拟净收益（元）": row.historical_profit_per_application}
        for arm, row in (("A：Champion", a), ("B：Challenger", b))])
    comparison_display = comparison.copy()
    comparison_display["threshold"] = comparison.threshold.map(lambda value: f"{value:.6f}" if pd.notna(value) else "n.a.")
    report = f"""# 审批策略模拟：固定阈值、现金流与风险约束

## 本阶段结果

使用已冻结的 Champion（原始 LightGBM）和 Challenger（三种子集成），没有新增模型训练或概率校准。
将训练 OOF 按标签分层分成策略开发 **{counts['policy_development']:,}** 行和策略评价 **{counts['policy_evaluation']:,}** 行。
只在开发子集拟合阈值、比较收益并选择 A/B 两套策略；评价子集和 {test_applications:,} 条测试申请均使用冻结阈值。
此划分仅隔离本阶段的策略选择。底层 OOF 已用于模型开发，模型之间还有共享训练数据，
因此评价结果仍是历史开发数据的离线模拟，不能称为全新独立模型测试或真实 A/B 结果。

冻结后的 A、B 策略：

{policy.business.markdown_table(definitions, rates=('target_approval_rate',))}

策略评价子集的基础情景结果：

{policy.business.markdown_table(evaluation_display, rates=('评价审批率', '获批历史违约率'), digits=2)}

在策略评价子集上，B 相对 A 的每个申请模拟净收益差为 **{delta:+.2f} 元**。
这不是实际收入增长或显著性结论，当前没有随机分流、Bootstrap 实验区间或功效分析。
按同一 {config['constraints']['max_development_default_rate']:.0%} 风险上限和 {config['constraints']['min_development_approval_rate']:.0%} 审批率下限检查，评价子集 A/B 通过状态为 **{feasible}**，只是点估计检查。

## 1. 明确的模拟经济假设

{policy.business.markdown_table(economics, digits=2)}

金额采用配置中的假设币种，期限统一为 {int(base.horizon_months)} 个月；匿名 Kaggle 数据没有实际本金、币种、合同利率或还款日期。
每条申请都假设相同本金，采用简化的一期现金流，不模拟摊还、提前还款和不同违约时间。
`label=1` 时取得全部设定利息；`label=0` 时没有利息收入，损失本金的 LGD 比例。
资金成本和服务成本发生在所有获批贷款，申请成本发生在全部申请，包括拒绝申请。
**收回的本金不计入收益**，也没有把违约标签解释为损失 100% 本金。

```text
获批且还款：本金 × 利息收益率 − 本金 × 资金成本率 − 服务成本 − 申请成本
获批且违约：−本金 × LGD − 本金 × 资金成本率 − 服务成本 − 申请成本
拒绝申请：−申请成本
每个申请净收益：全体申请净收益合计 / 全体申请人数
```

基础假设下，获批还款/违约/拒绝一笔的净收益分别为
**{base.principal * base.interest_rate_for_horizon - base.principal * base.funding_rate_for_horizon - base.servicing_cost_per_approved - base.application_cost_per_application:.0f} / {-base.principal * base.loss_given_default - base.principal * base.funding_rate_for_horizon - base.servicing_cost_per_approved - base.application_cost_per_application:.0f} / {-base.application_cost_per_application:.0f} 元**。
与拒绝相比的单笔审批增量盈亏平衡还款概率为 **{base.incremental_break_even_p_repay:.4f}**。
申请成本无论是否获批都产生，所以增量审批的盈亏平衡式不重复加入申请成本；这只是原始概率正确时的理论诊断。

## 2. 候选阈值、审批预算与选择规则

每个模型在策略开发分数上拟合 {len(config['target_approval_rates'])} 个目标审批预算。只按分数确定阈值，不用测试分数、评价标签或测试标签。
同分样本不按 ID 任意拆分；边界同分导致超预算时整组排除。因此真实审批人数可能低于目标。
审批规则为 `p_repay >= threshold`，另设全批/全拒两个端点。全拒策略的获批违约率为 NULL，人数为零，仍承担申请成本。

在基础情景、策略开发子集上，要求获批实际违约率 ≤ {config['constraints']['max_development_default_rate']:.0%}、审批率 ≥ {config['constraints']['min_development_approval_rate']:.0%}，
分别选择每个模型下每个申请历史模拟净收益最高的策略。利润按六位小数比较，
平手时依次选择违约率更低、审批率更低、policy_id 更早的方案。没有可行方案时明确失败，不强行选择。
这是历史标签点估计下的开发规则，不是统计风险保证或生产审批。

相同目标审批预算在评价子集上的对照如下；冻结阈值下两组评价审批率可能不同：

{policy.business.markdown_table(paired, rates=('target_approval_rate', 'delta_approval_rate', 'delta_default_rate'), digits=2)}

`exactly_equal_approved_count` 明确记录是否真正等量；差值均为 B 减 A，不把不同审批人群的坏账率差解释为因果改善。

## 3. 冻结策略的评价与测试应用

{policy.business.markdown_table(comparison_display, rates=('approval_rate', 'observed_default_rate_approved'), digits=2)}

历史模拟收益使用 OOF 标签；`raw_score_implied_profit_per_application` 使用未经校准的原始模型分数，单独列示。
测试真实违约率和历史模拟净收益为 NULL（n.a.）。测试分数可以计算假设条件下的模型隐含收益，
但不能报告为真实收益或已验证的未来利润。训练—测试漂移，以及 OOF/测试平均模型数量的差异仍然存在。

**审批量漂移是本次模拟的重要限制**：A、B 测试审批率分别仅为 **{test_a.approval_rate:.2%} / {test_b.approval_rate:.2%}**，
明显低于评价子集的 **{a.approval_rate:.2%} / {b.approval_rate:.2%}**。
按开发选择采用的 {config['constraints']['min_development_approval_rate']:.0%} 审批率下限检查，测试审批量通过状态为
**{ {arm: bool(row.approval_rate >= config['constraints']['min_development_approval_rate']) for arm, row in (('A', test_a), ('B', test_b))} }**。
这只是固定阈值下的审批量变化，不能推断测试真实坏账率。后续实验设计应先安排 shadow mode，
检查目标人群的分数分布、审批量和成熟标签风险，不能直接把 OOF 利润差作为上线收益预测。
此处保留原阈值，不利用测试分数事后调整配额。

![审批率、违约率与模拟收益](figures/approval_risk_profit.png)

## 4. 经济假设敏感性

分别改变利息、LGD、资金成本或本金；其他假设保持一致。全部情景使用同一套已经选好的 A/B 阈值，
不在不同情景重新挑策略，避免把事后最优方案混入固定策略对比。

{policy.business.markdown_table(sensitivity, digits=2)}

![经济敏感性](figures/economic_sensitivity.png)

## 5. 排序能力与风险捕获

ROC/PR 与 Lift/Gains 仅使用策略评价子集的 OOF；PR 以违约为少数正类，AP 为 Average Precision。
Gains 先按违约分数聚合同分，再从高风险向低风险累计。绘图抽稀不改变完整数据指标。

![ROC 与 PR](figures/roc_pr.png)

![Lift 与 Gains](figures/lift_gains.png)

## 6. 核查与后续实验交接

核查状态 **{audit['status']}**：SQL 聚合与逐申请 NumPy 现金流一致；本金收入、拒绝成本、零审批、
测试标签缺失、选择数据边界、同分预算、冻结阈值、配对差值、敏感性和逐申请 Parquet 均通过。
旧 SQL 分析库、冻结模型及历史训练记录以只读方式使用。

复现与字段说明：`docs/approval_policy_simulation.md`。
数据库：`data/interim/{output.run_name}.duckdb`；选定策略、开发/评价划分和逐申请收益在 `artifacts/{output.run_name}/`。
`experiment_policy_handoff.json` 给出后续实验设计使用的 A/B 规则和评价指标，尚未生成实验分组。

离线模拟额外假定：历史还款标签在不同审批决策与贷款条款下保持不变，数据覆盖拟讨论的申请人群。
真实审批可能改变人群、合同和客户行为；历史数据还可能存在已获批贷款的选择偏差。
没有借款日期和完整贷款生命周期，不能验证期限假设、逾期率、因果效应或生产风险。
下一步才能设计共同入组安全条件、随机化、样本量、结果成熟窗口和停止规则。
"""
    (output.reports / "approval_policy_simulation_report.md").write_text(report, encoding="utf-8")
    clean_definitions = definitions.astype(object).where(pd.notna(definitions), None).to_dict(orient="records")
    summary = {"status": "completed", "policy_cohort_rows": {k: int(v) for k, v in counts.items()},
        "selected_policies": clean_definitions,
        "evaluation": {"champion_approval_rate": float(a.approval_rate), "challenger_approval_rate": float(b.approval_rate),
            "champion_default_rate": float(a.observed_default_rate_approved), "challenger_default_rate": float(b.observed_default_rate_approved),
            "champion_profit_per_application": float(a.historical_profit_per_application), "challenger_profit_per_application": float(b.historical_profit_per_application),
            "delta_profit_per_application": delta, "point_estimate_constraints_passed": feasible},
        "economics": config["economics"], "probability_calibrated": False, "randomized_experiment_completed": False,
        "unlabeled_test": {"applications": test_applications, "champion_approval_rate": float(test_a.approval_rate),
            "challenger_approval_rate": float(test_b.approval_rate), "observed_default_rates_available": False,
            "historical_simulated_profits_available": False,
            "point_estimate_min_approval_rate_passed": {arm: bool(row.approval_rate >= config['constraints']['min_development_approval_rate']) for arm, row in (("A", test_a), ("B", test_b))}},
        "evaluation_scope": "policy_evaluation_subset_of_reused_development_oof"}
    policy.business.write_json(output.reports / "summary.json", summary)
    policy.business.write_json(output.artifacts / "experiment_policy_handoff.json", {
        **summary, "handoff_status": "candidate_rules_for_experiment_design" if all(feasible.values()) else "evaluation_risk_constraint_failed_do_not_advance",
        "policy_split_file": policy.business.relative(output.artifacts / "policy_cohorts.parquet", output.root),
        "application_outcomes_file": policy.business.relative(output.artifacts / "selected_policy_application_outcomes.parquet", output.root),
        "primary_metric_proposal": "net_profit_per_randomized_application_ITT",
        "production_rollout_ready": False,
        "shadow_validation_required": True,
        "default_rate_constraint_is_point_estimate": True,
        "required_next_steps": ["shadow_mode_score_drift_and_approval_volume_validation", "common_pre_assignment_safety_eligibility", "randomization_design", "sample_size_and_MDE", "outcome_maturity_and_stopping_rules"],
        "offline_assumption": config["offline_outcome_assumption"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="approval_policy_simulation")
    args = parser.parse_args()
    output, config = policy.paths(args.run_name), policy.load_config()
    start = time.perf_counter()
    print("Checking the completed SQL business layer (read-only)", flush=True)
    verify_parent(policy.business.Paths(output.root, config["parent_run_name"]))
    predictions, split, candidates = policy.prepare_inputs(output, config)
    print(f"Policy cohorts: {split.policy_cohort.value_counts().to_dict()}; {len(candidates)} candidate policies", flush=True)
    selected = build_database(output, config, candidates)
    policy.freeze_definitions(output, config, selected)
    export_results(output)
    files = [Path(__file__), policy.CONFIG, *policy.SQL_FILES, policy.ROOT / "scripts/policy_simulation_common.py",
             policy.ROOT / "scripts/verify_approval_policy_simulation.py"]
    run = {"status": "verifying", "run_name": output.run_name, "duckdb_version": duckdb.__version__,
        "analysis_code_sha256": {policy.business.relative(p, output.root): policy.business.sha256(p) for p in files},
        "policy_manifest_sha256": policy.business.sha256(output.artifacts / "policy_freeze_manifest.json"), "config": config}
    policy.business.write_json(output.reports / "run_config.json", run)
    audit = verify(output)
    policy.business.write_json(output.reports / "artifact_verification.json", audit)
    render_figures(output, config, predictions, split)
    write_report(output, config, selected, split, audit)
    run.update(status="completed", elapsed_seconds=time.perf_counter() - start,
               database_sha256=policy.business.sha256(output.database),
               application_outcomes_sha256=policy.business.sha256(output.artifacts / "selected_policy_application_outcomes.parquet"))
    policy.business.write_json(output.reports / "run_config.json", run)
    print(f"Completed and verified in {run['elapsed_seconds']:.1f}s: {output.reports / 'approval_policy_simulation_report.md'}", flush=True)


if __name__ == "__main__": main()
