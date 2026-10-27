"""Plot completed controls and generate a Chinese beginner-readable report."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import run_tree_model_comparison as study
except ImportError:
    import run_tree_model_comparison as study


NAMES = {"lightgbm_baseline": "LightGBM baseline", "lightgbm_tuned_seed_42": "LightGBM tuned seed 42",
         "lightgbm_tuned_ensemble": "LightGBM 3-seed mean", "xgboost": "XGBoost control", "catboost": "CatBoost control"}
COLORS = ["#a8adb5", "#719eca", "#214f82", "#df9149", "#638b6d"]


def table(frame):
    frame = frame.copy()
    for column in frame.select_dtypes(include="floating").columns:
        frame[column] = frame[column].map(lambda value: f"{value:.6f}")
    return study.base.markdown_table(frame)


def figures(overall, paired, selection):
    folder = study.REPORT / "figures"
    folder.mkdir(exist_ok=True)
    models = list(NAMES)
    frame = overall.set_index("model").loc[models]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, study.METRICS, ["OOF ROC-AUC (higher is better)", "Default average precision (higher is better)",
                                                             "Log loss (lower is better)", "Brier score (lower is better)"]):
        values = frame[metric].to_numpy()
        position = np.arange(len(models))
        ax.barh(position, values, color=COLORS, height=.64)
        ax.set_yticks(position, [NAMES[model] for model in models], fontsize=9)
        ax.invert_yaxis()
        lo, hi = float(values.min()), float(values.max())
        span = max(hi - lo, abs(hi) * .015)
        ax.set_xlim(lo - .3 * span, hi + 1.1 * span)
        for row, value in enumerate(values):
            ax.text(value + .05 * span, row, f"{value:.6f}", va="center", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Same development folds and 1,105 numeric features; search/model counts differ", fontsize=12)
    fig.savefig(folder / "overall_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for engine, color in zip(study.ENGINES, COLORS[-2:]):
        row = paired.query("model == @engine").sort_values("fold")
        axes[0].plot(row.fold + 1, row.delta_single_repay_roc_auc, marker="o", color=color, label=NAMES[engine])
        axes[1].plot(row.fold + 1, row.delta_ensemble_repay_roc_auc, marker="o", color=color, label=NAMES[engine])
    for ax, title in zip(axes, ["AUC difference vs tuned LightGBM seed 42", "AUC difference vs LightGBM 3-seed mean"]):
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Validation fold (plot 1-5 = code 0-4)")
        ax.set_ylabel("Paired AUC difference")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.savefig(folder / "paired_fold_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for engine, color in zip(study.ENGINES, COLORS[-2:]):
        for fold in range(5):
            curve = json.loads((study.REPORT / "inner_curves" / f"{engine}_fold_{fold}.json").read_text())["auc"]
            chosen = selection.query("engine == @engine and fold == @fold").iloc[0]
            ax.plot(np.arange(1, len(curve) + 1), curve, color=color, alpha=.3, linewidth=.9)
            ax.scatter(chosen.selected_rounds, chosen.inner_auc, color=color, s=24, zorder=3)
        ax.plot([], [], color=color, label=NAMES[engine])
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Inner early-stop AUC")
    ax.set_title("Five inner curves per engine; dots are retained rounds, not outer scores")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(folder / "inner_learning_curves.png", dpi=170)
    plt.close(fig)


def report():
    config = study.checked_config()
    if config["status"] != "completed":
        raise RuntimeError("Training must be completed")
    experiment = config["experiment"]
    summary = json.loads((study.REPORT / "summary.json").read_text())
    audit = json.loads((study.REPORT / "artifact_verification.json").read_text())
    assert audit["status"] == "passed" and audit["training_signature"] == config["training_signature"]
    comparison = pd.read_csv(study.REPORT / "comparison_metrics.csv")
    paired = pd.read_csv(study.REPORT / "paired_fold_metrics.csv")
    selection = pd.read_csv(study.REPORT / "selection_records.csv")
    records = pd.read_csv(study.REPORT / "refit_records.csv")
    correlations = pd.read_csv(study.REPORT / "prediction_correlations.csv", index_col=0)
    overall = comparison.query("group == 'overall'")
    figures(overall, paired, selection)
    indexed = overall.set_index("model")
    single, incumbent = indexed.loc["lightgbm_tuned_seed_42"], indexed.loc["lightgbm_tuned_ensemble"]
    difference = pd.DataFrame([{"model": engine,
        **{f"vs_single_{key}": float(indexed.loc[engine, key] - single[key]) for key in study.METRICS},
        **{f"vs_ensemble_{key}": float(indexed.loc[engine, key] - incumbent[key]) for key in study.METRICS}}
        for engine in study.ENGINES])
    costs = pd.DataFrame([{"engine": engine, "inner_models": 5, "refit_models": 5,
                          "selected_trees_sum": int(records.query("engine == @engine").refit_rounds.sum()),
                          "model_stage_seconds_sum": float(selection.query("engine == @engine").elapsed_seconds.sum() + records.query("engine == @engine").elapsed_seconds.sum()),
                          "final_model_bytes": sum(study.model_paths(engine, fold, "refit")[0].stat().st_size for fold in range(5))}
                         for engine in study.ENGINES])
    execution_rows = [{"engine": row["engine"], "stage": stage,
                       "mode": row.get("execution_mode", "single"),
                       "threads": row.get("training_threads", experiment["threads"])}
                      for stage, frame in (("inner", selection), ("refit", records))
                      for row in frame.to_dict(orient="records")]
    execution_table = pd.DataFrame(execution_rows).groupby(
        ["engine", "stage", "mode", "threads"], sort=True
    ).size().reset_index(name="model_count")
    pressure = comparison.query("group != 'overall'")[["model", "group", "rows", "default_rate", *study.METRICS]].copy()
    pressure.default_rate = pressure.default_rate.map(lambda value: f"{value:.4%}")
    recommendations = {"lightgbm_tuned_ensemble": "继续使用已有 LightGBM 三种子集成作为下一步开发候选。",
                       "xgboost": "XGBoost 对照满足预声明替换规则，作为下一步开发候选。",
                       "catboost": "CatBoost 对照满足预声明替换规则，作为下一步开发候选。"}
    notes = f"""# 第三步：XGBoost / CatBoost 对照

使用 **{experiment['train_rows']:,}** 条训练记录、**{experiment['test_rows']:,}** 条测试记录，
全部 **{len(experiment['features']):,}** 个原始 float32 数值特征。
`label=1` 表示还款，输出及提交均为还款概率，违约概率是 `1-p_repay`。
完成两个库各五个内部模型、五个完整四折重训模型，共 **20 个新模型**。

{recommendations[summary['recommended_development_model']]}
本轮为具体配置的开发对照，尚无官方测试成绩，两个提交均未上传。

## 一、完整 OOF 比较

{table(overall[['model', *study.METRICS]])}

AUC 与违约 AP 越高越好，log loss 与 Brier 越低越好。
AP 使用 `average_precision_score(1-label, 1-p_repay)`。
LightGBM 单种子和两个新库各有 5 个最终折模型，已有 LightGBM 集成有 15 个。
LightGBM 之前选过四组参数，本轮两个库各只有一组预声明参数，搜索预算不同。
不能将这份表解释为全局最优算法排名。

{table(difference[['model', 'vs_single_repay_roc_auc', 'vs_single_default_pr_auc_ap', 'vs_ensemble_repay_roc_auc', 'vs_ensemble_default_pr_auc_ap']])}

![整体比较]({(study.REPORT / 'figures/overall_comparison.png').as_posix()})

## 二、验证协议和参数

已有分层五折保持不变。留出一折，其余四折内部按标签分层划为 85% 训练、15% 早停。
分箱只基于内部训练行，早停行复用训练边界。内部 AUC 选择树数后，在完整拟合四折上
从头重训，再预测外层验证和测试。每个库基础种子 42，实际为 `42+fold`。
最多 {experiment['max_rounds']} 轮，耐心 {experiment['patience']} 轮，有效改善要求超过 {experiment['min_delta']}。

XGBoost：CPU hist、depthwise、学习率 0.05、深度 6、min_child_weight 20、L2 5、
行/列抽样各 0.8、max_bin 63。min_child_weight 约束 Hessian 之和，不是样本行数。

CatBoost：CPU Plain、SymmetricTree、学习率 0.05、深度 6、L2 5、Bernoulli 行抽样 0.8、
rsm 0.8、random_strength 1、border_count 63、nan_mode Min、
boost_from_average True、leaf_estimation_iterations 10。
本轮没有显式类别特征，没有使用类别目标编码或 Ordered boosting。

## 三、实际选择的树数与预算

{table(selection[['engine', 'fold', 'selected_rounds', 'inner_auc', 'evaluated_rounds', 'hit_round_cap']])}

达到训练轮数上限的次数：XGBoost **{summary['hit_round_cap']['xgboost']}/5**，
CatBoost **{summary['hit_round_cap']['catboost']}/5**。
用完预算只能说明训练受到上限限制，不能说明树数已最优。
XGBoost best_iteration 是从 0 开始的索引；实际树数加 1。
CatBoost 使用自定义同规则回调，再 shrink 到选中轮数，避免与库的默认最佳轮数规则混用。

![内部学习曲线]({(study.REPORT / 'figures/inner_learning_curves.png').as_posix()})

## 四、逐折对照

{table(paired.query('model in @study.ENGINES')[['model', 'fold', 'repay_roc_auc', 'delta_single_repay_roc_auc', 'delta_ensemble_repay_roc_auc', 'default_pr_auc_ap', 'log_loss']])}

高于调参 LightGBM 单种子 AUC 的验证折：XGBoost **{summary['folds_above_tuned_single_auc']['xgboost']}/5**，
CatBoost **{summary['folds_above_tuned_single_auc']['catboost']}/5**。
代码/表使用 0–4 折号，图使用 1–5，对应同一划分。
拟合四折之间有重叠，不能把五折当作五次独立实验计算正式置信区间。

![逐折差值]({(study.REPORT / 'figures/paired_fold_comparison.png').as_posix()})

## 五、既有压力组和模型差异

{table(pressure)}

两个压力组有重叠，不能相加；AP 受每组违约率影响。
这些组不参与本轮树数选择，测试 feature_492 的未见范围漂移仍未得到验证解决。

{table(correlations.loc[list(study.ENGINES), ['lightgbm_tuned_seed_42', 'lightgbm_tuned_ensemble', 'xgboost', 'catboost']].reset_index(names='model'))}

上表是预测概率的 Pearson 相关性，不是直接测量错误相关性。
不同预测可能给后续混合提供信息，但本轮没有按外层分数搜索或生成跨库集成。

## 六、资源与可复现产物

{table(costs)}

上表耗时是内部选择与重训阶段之和，包含数据构建、CatBoost TSV 导出、量化和预测，
不是只计 boosting，也不是全项目历时或部署延迟。最终模型字节数不等于运行内存。
实际模型阶段的执行模式与线程数如下；多进程调度最多两个进程、合计 10 个逻辑 CPU。
只有一个未完成任务时为一个进程、10 线程。量化仍需内存，资源请求不等于进程硬上限。

{table(execution_table)}

XGBoost 的 DataIter / QuantileDMatrix 分批量化，早停使用 ref。
原始特征由 BatchedFeatureFile 每批打开、复制后关闭只读映射，降低原始缓存与量化数据同时驻留的压力。
CatBoost 从分批输出的临时 TSV 直接量化，早停使用内部训练 borders。
训练没有边界的列在早停量化时显式忽略，阻止由早停数据新建边界。
保存出的早停/训练边界完全一致；临时 TSV 只在量化过程中存在。
9 位有效数字保持 float32 往返一致，NaN 和真实零保留。
原始 DOUBLE 已经发生的 float32 舍入不会由文本或 float64 转换恢复。

模型保存于 `{study.ARTIFACTS.as_posix()}/inner_models`、`models`；
原始行索引与保存概率在 `inner_splits`、`inner_predictions`、`fold_predictions`。
OOF 和测试汇总为 Parquet；两个本地提交是
`{study.submission_path('xgboost').as_posix()}`、`{study.submission_path('catboost').as_posix()}`，按官方模板 id 对齐。

## 七、核查与开发结论

重载 **{audit['inner_models_reloaded']} 个内部模型、{audit['refit_models_reloaded']} 个最终模型**。
重放早停、复现并检查无交叉划分、核对模型/预测/引用校验值、特征名、树数、
每种模型 OOF 恰好覆盖一次、所有指标/相关性/推荐规则及两个提交。
CatBoost 训练/早停边界一致，已核查 {audit['catboost_border_files_verified']} 个边界文件。
内部、OOF、测试重载预测最大差分别为 **{audit['max_inner_prediction_difference']}**、
**{audit['max_oof_difference']}**、**{audit['max_test_difference']}**。

预声明替换规则：新模型相对已有 LightGBM 三种子集成，AUC 至少提高 0.0005，
违约 AP 下降不超过 0.001，log loss 增加不超过 0.0005。
{recommendations[summary['recommended_development_model']]}
这是开发选择阈值，不是显著性检验或业务部署批准。

项目前几轮已使用这些折进行开发，因此这是开发 OOF 而非从未查看过的最终测试。
尚未做官方提交、概率校准、成本阈值优化、可靠时间外推或生产部署。
本轮低分只能描述给定配置和预算；换参数、增加预算可能改变结果。
代码逐段说明、复现命令和面试回答见 `docs/xgboost_catboost_comparison.md`。
"""
    (study.REPORT / "tree_model_comparison_report.md").write_text(notes, encoding="utf-8")
    print("Generated audited report and three figures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="tree_model_comparison")
    args = parser.parse_args()
    study.configure_run(args.run_name)
    report()
