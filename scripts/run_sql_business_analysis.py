"""Freeze existing models, build SQL business tables, verify and report results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import duckdb
import pandas as pd

try:
    from . import business_analysis_common as common
    from .verify_sql_business_analysis import verify
except ImportError:
    import business_analysis_common as common
    from verify_sql_business_analysis import verify


def build_database(paths, config):
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.database.with_name(f".{paths.run_name}_{time.time_ns()}.duckdb")
    temp_directory = paths.artifacts / "duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    try:
        with duckdb.connect(str(temporary)) as connection:
            connection.execute("SET threads = ?", [config["threads"]]).fetchall()
            connection.execute("SET memory_limit = ?", [config["memory_limit"]]).fetchall()
            connection.execute("SET temp_directory = ?", [str(temp_directory)]).fetchall()
            connection.execute("BEGIN TRANSACTION").fetchall()
            common.execute_analysis_sql(connection, paths.predictions, paths.registry, config["approval_thresholds"])
            common.copy_quality_tables(connection, paths.root / "data/interim/loan_repayment.duckdb")
            connection.execute("COMMIT").fetchall()
            connection.execute("CHECKPOINT").fetchall()
        temporary.replace(paths.database)
    finally:
        if temporary.exists(): temporary.unlink()


def export_tables(paths):
    output = paths.reports / "tables"
    output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(paths.database), read_only=True) as connection:
        for table in common.EXPORT_TABLES:
            connection.execute(f"COPY {table} TO ? (FORMAT CSV, HEADER)", [str(output / f"{table}.csv")]).fetchall()


def render_figures(paths):
    # Keep Matplotlib's writable cache inside this stage's workspace.
    cache = paths.artifacts / "matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = paths.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(paths.database), read_only=True) as connection:
        bands = connection.table("risk_band_summary").fetchdf()
        policies = connection.table("approval_policy_results").fetchdf()
    colors = {"champion": "#3266a8", "challenger": "#d47b23"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for role, color in colors.items():
        train = bands.query("dataset == 'train' and model_role == @role").sort_values("risk_decile")
        test = bands.query("dataset == 'test' and model_role == @role").sort_values("risk_decile")
        axes[0].plot(train.risk_decile, train.observed_default_rate * 100, marker="o", color=color, label=f"{role}: observed")
        axes[0].plot(train.risk_decile, train.mean_model_p_default * 100, linestyle="--", color=color, label=f"{role}: raw score")
        axes[1].plot(test.risk_decile, test.population_share * 100, marker="o", color=color, label=role)
    axes[0].set(title="Development OOF: default rate by risk band", xlabel="Risk decile (10 = highest risk)", ylabel="Default rate / raw score (%)")
    axes[1].axhline(10, color="gray", linestyle=":", label="10% reference")
    axes[1].set(title="Test occupancy using train-OOF boundaries", xlabel="Risk decile (10 = highest risk)", ylabel="Share of test rows (%)")
    for axis in axes:
        axis.set_xticks(range(1, 11)); axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure.savefig(figures / "risk_deciles.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for role, color in colors.items():
        train = policies.query("dataset == 'train' and model_role == @role").sort_values("threshold")
        test = policies.query("dataset == 'test' and model_role == @role").sort_values("threshold")
        axes[0].plot(train.approval_rate * 100, train.observed_default_rate_approved * 100, marker="o", color=color, label=role)
        axes[1].plot(train.threshold, train.approval_rate * 100, color=color, label=f"{role}: OOF")
        axes[1].plot(test.threshold, test.approval_rate * 100, linestyle="--", color=color, label=f"{role}: test")
    axes[0].set(title="Exploratory thresholds: development OOF", xlabel="Approval rate (%)", ylabel="Observed default rate among approved (%)")
    axes[1].set(title="Fixed threshold grid; no policy selected", xlabel="Repayment probability threshold", ylabel="Approval rate (%)")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure.savefig(figures / "approval_thresholds.png", dpi=160)
    plt.close(figure)


def write_report(paths, audit):
    with duckdb.connect(str(paths.database), read_only=True) as connection:
        metrics = connection.execute("SELECT model_role, repay_roc_auc, default_pr_auc_ap, ks, log_loss, brier_score FROM model_performance_summary ORDER BY model_role").fetchdf()
        top = connection.execute("SELECT model_role, row_count, observed_default_rate, default_lift, default_capture_share FROM risk_band_summary WHERE dataset = 'train' AND risk_decile = 10 ORDER BY model_role").fetchdf()
        distribution = connection.execute("SELECT dataset, model_role, mean_p_repay, p10, p50, p90, observed_default_rate FROM prediction_score_summary ORDER BY model_role, dataset").fetchdf()
        drift = connection.execute("SELECT model_role, score_psi, decile_total_variation FROM prediction_drift_summary ORDER BY model_role").fetchdf()
        thresholds = connection.execute("SELECT model_role, threshold, approval_rate, observed_default_rate_approved, mean_model_p_default_approved FROM approval_policy_results WHERE dataset = 'train' AND threshold IN (0.8, 0.9, 0.95) ORDER BY threshold, model_role").fetchdf()
        row_count = connection.execute("SELECT count(*) FROM model_predictions").fetchall()[0][0]
    by_role = metrics.set_index("model_role")
    delta = by_role.loc["challenger", "repay_roc_auc"] - by_role.loc["champion", "repay_roc_auc"]
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    text = f"""# SQL 业务分析：固定模型、风险分层与阈值描述

## 本阶段交付

冻结原始 LightGBM（Champion，5 个折模型）和调参三种子集成（Challenger，15 个折模型）。
688,998 条训练记录采用 OOF，36,324 条测试记录采用既有折模型平均；统一长表共 **{row_count:,}** 行。
没有新增模型训练、选择种子、重新推理、概率校准或选择最终审批策略。
Champion 是项目参照，不是已部署系统；两套模型均未上传 Kaggle，本报告没有官方测试成绩。

## 1. SQL 重算模型指标

{common.markdown_table(metrics)}

Challenger 相对 Champion 的开发 OOF AUC 差为 **{delta:+.6f}**。
AUC、违约 AP、KS 均先聚合同分样本，再用 SQL 窗口累计计算；log loss 和 Brier 用 SQL 聚合。
违约 AP 定义为 Average Precision，以 `1-label` 和 `1-p_repay` 计算；log loss 概率截断为 1e-15。
独立 NumPy/sklearn 重算通过。KS 是模型分数对两类标签的区分度，与此前用于特征漂移的两样本 KS 不同。
现有折已被多轮开发重复使用，这些指标属于开发 OOF，不是独立最终验证或显著性结论。

## 2. 风险十分位与高风险识别

每个模型用训练 OOF 的违约分数 Q10–Q90 定义边界；D1 风险最低，D10 风险最高。
边界原样应用于测试集，测试分数不参与拟合边界。同分样本不拆开，空风险层仍输出。
分层是相对模型风险，没有虚构收入、年龄、地区等匿名特征含义。

{common.markdown_table(top, rates=('observed_default_rate', 'default_capture_share'))}

`default_lift` = 层内实际违约率 / 全训练实际违约率；`default_capture_share` = 层内违约人数 / 全训练违约人数。
完整十分位表还记录还款率、还款提升度、平均原始违约分数，以及从高风险到低风险累计捕获的违约比例。
原始模型概率未校准，分数均值与实际违约率分别展示。

![风险分层与测试占比](figures/risk_deciles.png)

## 3. 训练与测试分数分布

{common.markdown_table(distribution, rates=('observed_default_rate',))}

{common.markdown_table(drift)}

测试没有标签，实际违约率、实际还款率、违约人数和捕获率均为 NULL（表中 n.a.），不是零。
分数 PSI 使用同一套训练 OOF 十分位，零占比用 1e-6 下限处理。
分数差异既可能包含数据漂移，也可能包含平均模型数量的差异：Champion OOF 每行用 1 个模型、测试用 5 个；
Challenger OOF 每行用 3 个模型、测试用 15 个。因此 PSI 只用于诊断，不能推断测试标签分布或真实风险。

## 4. 固定阈值网格的描述性分析

配置提前固定 13 个还款概率阈值，审批规则是 `p_repay >= threshold`，等于阈值时获批。
下表的 0.8、0.9、0.95 仅用于展示，其余阈值见完整 CSV；本阶段没有按结果挑选阈值。

{common.markdown_table(thresholds, rates=('approval_rate', 'observed_default_rate_approved', 'mean_model_p_default_approved'))}

审批率分母是全部申请；获批实际违约率分母是具有标签的获批申请。无人获批时实际违约率为 NULL。
测试只能输出审批占比和获批平均模型分数。`champion_challenger_comparison` 比较相同概率阈值，
两组审批率可能不同，不能把获批违约率差直接解释成同一人群的因果改善。

![阈值描述性分析](figures/approval_thresholds.png)

## 5. 数据来源和核查

新数据库：`data/interim/{paths.run_name}.duckdb`。旧数据质量数据库以只读方式读取，复用其物理质量表与漂移表。
冻结清单：`artifacts/{paths.run_name}/model_freeze_manifest.json`，使用项目相对路径和 SHA256，保留历史记录的原路径。
已检查 20 个完整折模型的文件校验值，以及折预测重建、OOF 覆盖、三种子平均、ID/标签/折号一致性。
本阶段没有重新运行原始特征的模型推理；历史原生模型重载核查记录保留，当前核查范围为：
**{manifest['verification_scope']}**。

独立核查状态：**{audit['status']}**。冻结输入和输出校验值、SQL 指标、分层、阈值分母、缺失标签、
测试共用边界、分数 PSI、配对差值和 11 份 CSV 全部通过。
复现命令与字段字典见 `docs/sql_business_analysis.md` 和 `docs/sql_business_data_dictionary.md`。

## 6. 完成边界

本阶段完成模型冻结和 SQL 业务分析。审批收益假设、风险约束下的最终策略选择、
A/B 分流、功效分析、实验周期及离线实验模拟留待后续。
后续实验需明确历史标签在不同审批策略下保持不变等离线假设，不能把本报告写成真实随机实验成果。
"""
    (paths.reports / "sql_business_analysis_report.md").write_text(text, encoding="utf-8")
    common.write_json(paths.reports / "summary.json", {"status": "completed", "prediction_rows": row_count,
        "overall_metrics": metrics.to_dict(orient="records"), "challenger_auc_delta": float(delta),
        "test_labels_available": False, "probability_calibrated": False, "approval_policy_selected": False,
        "economic_simulation_completed": False, "randomized_experiment_completed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="sql_business_analysis")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    paths, config = common.Paths(run_name=args.run_name), common.load_config()
    start = time.perf_counter()
    common.freeze_models(paths, config)
    if args.freeze_only:
        print(f"Model freeze complete: {paths.manifest}", flush=True)
        return
    print("Building business database using frozen predictions and SQL", flush=True)
    build_database(paths, config)
    export_tables(paths)
    # Record the current analysis code before verification, including reruns.
    code_files = [common.CONFIG, *common.SQL_FILES, Path(__file__),
                  common.ROOT / "scripts/business_analysis_common.py", common.ROOT / "scripts/verify_sql_business_analysis.py"]
    run = {"status": "verifying", "run_name": paths.run_name, "duckdb_version": duckdb.__version__,
        "analysis_code_sha256": {common.relative(p, paths.root): common.sha256(p) for p in code_files},
        "freeze_manifest_sha256": common.sha256(paths.manifest), "config": config}
    common.write_json(paths.reports / "run_config.json", run)
    audit = verify(paths)
    common.write_json(paths.reports / "artifact_verification.json", audit)
    render_figures(paths)
    write_report(paths, audit)
    run.update(status="completed", elapsed_seconds=time.perf_counter() - start,
               database_sha256=common.sha256(paths.database))
    common.write_json(paths.reports / "run_config.json", run)
    print(f"Completed and verified in {run['elapsed_seconds']:.1f}s: {paths.reports / 'sql_business_analysis_report.md'}", flush=True)


if __name__ == "__main__": main()
