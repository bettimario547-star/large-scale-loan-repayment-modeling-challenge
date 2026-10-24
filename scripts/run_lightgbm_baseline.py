"""Full-data LightGBM baseline with untouched outer folds and inner early stopping.

Select a tree count on an inner split, then refit on all four outer fitting folds.
LightGBM Sequence reads the existing raw float32 cache in bounded batches; NaNs
remain NaNs. Binning is fitted only on the fitting rows for each training stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

try:
    from .id_feature_contract import align_validation_folds, build_submission, model_feature_columns
    from .run_linear_baselines import (
        ROOT, TRAIN, TEST, FOLDS, ARTIFACTS as LINEAR_ARTIFACTS,
        blocks, json_write, markdown_table, metrics, raw_feature_cache, source_signature,
    )
except ImportError:
    from id_feature_contract import align_validation_folds, build_submission, model_feature_columns
    from run_linear_baselines import (
        ROOT, TRAIN, TEST, FOLDS, ARTIFACTS as LINEAR_ARTIFACTS,
        blocks, json_write, markdown_table, metrics, raw_feature_cache, source_signature,
    )


REPORT = ROOT / "reports/lightgbm_baseline"
ARTIFACTS = ROOT / "artifacts/lightgbm_baseline"
SEED = 42


class IndexedSequence(lgb.Sequence):
    """Expose selected raw rows without copying the complete fitting matrix."""

    def __init__(self, values: np.ndarray, indices: np.ndarray, batch_rows: int = 4096):
        self.values = values
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_rows
        if values.ndim != 2 or self.indices.ndim != 1 or batch_rows < 1:
            raise ValueError("Expected a matrix, one-dimensional row indices, and positive batch size")
        if len(self.indices) == 0 or np.any((self.indices < 0) | (self.indices >= len(values))):
            raise ValueError("Sequence indices must select nonempty, valid rows")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item) -> np.ndarray:
        # LightGBM's sampled-column and push-row APIs require float64. Only
        # the current row/batch is converted; NaN and actual zero are preserved.
        rows = self.indices[item]
        result = np.array(self.values[rows], dtype=np.float64, order="C", copy=True)
        if np.isinf(result).any():
            raise ValueError("Infinite feature values are not supported")
        return result


def split_outer_fit(labels: np.ndarray, fold_id: np.ndarray, fold: int,
                    early_stop_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(labels) != len(fold_id) or not 0 < early_stop_fraction < 1:
        raise ValueError("Invalid labels, folds, or early-stop fraction")
    candidates = np.flatnonzero(fold_id != fold)
    outer_valid = np.flatnonzero(fold_id == fold)
    inner_fit, early_stop = train_test_split(
        candidates, test_size=early_stop_fraction, stratify=labels[candidates],
        random_state=SEED + fold,
    )
    return np.sort(inner_fit), np.sort(early_stop), outer_valid


def model_params(fold: int, threads: int) -> dict:
    return {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
        "min_data_in_leaf": 200, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "max_bin": 63, "bin_construct_sample_cnt": 20_000,
        "use_missing": True, "zero_as_missing": False,
        "num_threads": threads, "deterministic": True, "force_col_wise": True,
        "histogram_pool_size": 128, "verbosity": -1,
        "seed": SEED + fold, "data_random_seed": SEED + fold,
        "feature_fraction_seed": SEED + fold, "bagging_seed": SEED + fold,
    }


def make_dataset(values: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                 names: list[str], params: dict, batch_rows: int,
                 reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(
        IndexedSequence(values, indices, batch_rows), label=labels[indices],
        feature_name=names, params=params, reference=reference, free_raw_data=True,
    )


def predict_rows(model: lgb.Booster, values: np.ndarray, batch_rows: int,
                 indices: np.ndarray | None = None) -> np.ndarray:
    rows = np.arange(len(values)) if indices is None else np.asarray(indices)
    predictions = np.empty(len(rows), dtype=np.float64)
    for section in blocks(len(rows), batch_rows):
        chunk = np.array(values[rows[section]], dtype=np.float32, order="C", copy=True)
        predictions[section] = model.predict(chunk, num_iteration=model.current_iteration())
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise RuntimeError("Invalid repayment probabilities")
    return predictions


def log_refit(prefix: str):
    start = time.perf_counter()

    def callback(environment):
        iteration = environment.iteration + 1
        if iteration % 50 == 0:
            print(f"{prefix}: trees={iteration}, elapsed={time.perf_counter() - start:.0f}s", flush=True)

    callback.order = 20
    return callback


def run_fold(values: np.ndarray, test_values: np.ndarray, labels: np.ndarray,
             fold_id: np.ndarray, names: list[str], fold: int, args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    inner_fit, early_stop, outer_valid = split_outer_fit(labels, fold_id, fold, args.early_stop_fraction)
    all_fit = np.flatnonzero(fold_id != fold)
    params = model_params(fold, args.threads)
    np.savez_compressed(ARTIFACTS / f"inner_splits/fold_{fold}.npz",
                        inner_fit_index=inner_fit, early_stop_index=early_stop,
                        outer_valid_index=outer_valid)
    prefix = f"Fold {fold + 1}/5"
    print(f"{prefix}: inner fit={len(inner_fit):,}, early stop={len(early_stop):,}, "
          f"outer valid={len(outer_valid):,}", flush=True)
    inner_data = make_dataset(values, labels, inner_fit, names, params, args.batch_rows)
    stop_data = make_dataset(values, labels, early_stop, names, params, args.batch_rows, reference=inner_data)
    history = {}
    selected = lgb.train(
        params, inner_data, num_boost_round=args.max_rounds,
        valid_sets=[stop_data], valid_names=["inner_early_stop"],
        callbacks=[lgb.early_stopping(args.patience, first_metric_only=True, min_delta=args.min_delta),
                   lgb.log_evaluation(50), lgb.record_evaluation(history)],
    )
    selected_rounds = int(selected.best_iteration)
    if selected_rounds < 1:
        raise RuntimeError("Early stopping did not produce a valid tree count")
    selection_seconds = time.perf_counter() - start
    selection = {"selected_rounds": selected_rounds,
                 "inner_early_stop_auc": float(selected.best_score["inner_early_stop"]["auc"]),
                 "selection_evaluated_rounds": len(history["inner_early_stop"]["auc"]),
                 "selection_hit_round_cap": len(history["inner_early_stop"]["auc"]) >= args.max_rounds,
                 "selection_seconds": selection_seconds}
    json_write(REPORT / f"learning_curves/fold_{fold}.json", history)
    print(f"{prefix}: selected {selected_rounds} trees. Refit on all {len(all_fit):,} outer fitting rows.", flush=True)
    del selected, inner_data, stop_data
    gc.collect()
    full_data = make_dataset(values, labels, all_fit, names, params, args.batch_rows)
    # The tree count is already chosen. Outer validation is not passed to train.
    model = lgb.train(params, full_data, num_boost_round=selected_rounds,
                      callbacks=[log_refit(f"{prefix} refit")])
    del full_data
    gc.collect()
    model_path = ARTIFACTS / f"models/fold_{fold}.txt"
    model.save_model(str(model_path), num_iteration=model.current_iteration())
    valid_score = predict_rows(model, values, args.batch_rows, outer_valid)
    test_score = predict_rows(model, test_values, args.batch_rows)
    gain = model.feature_importance(importance_type="gain")
    share = gain / gain.sum() if gain.sum() else np.zeros_like(gain)
    split = model.feature_importance(importance_type="split")
    pd.DataFrame({"fold": fold, "feature": names, "gain": gain, "gain_share": share,
                  "split_count": split}).to_csv(REPORT / f"feature_importance/fold_{fold}.csv", index=False)
    record = {"model": "lightgbm", "fold": fold, "outer_fit_rows": len(all_fit),
              "inner_fit_rows": len(inner_fit), "early_stop_rows": len(early_stop),
              "refit_rounds": model.current_iteration(), **selection,
              **metrics(labels[outer_valid], valid_score), "elapsed_seconds": time.perf_counter() - start}
    json_write(ARTIFACTS / f"models/fold_{fold}.json", {"features": names, "positive_class": 1,
               "params": params, "record": record, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()})
    np.savez_compressed(ARTIFACTS / f"fold_predictions/fold_{fold}.npz",
                        outer_valid_index=outer_valid, p_repay_valid=valid_score, p_repay_test=test_score)
    print(f"{prefix}: outer AUC={record['repay_roc_auc']:.6f}, default AP={record['default_pr_auc_ap']:.6f}, "
          f"log loss={record['log_loss']:.6f}", flush=True)
    del model
    gc.collect()
    return record


def comparison_metrics(train_meta: pd.DataFrame, folds: pd.DataFrame, lightgbm_oof: np.ndarray) -> pd.DataFrame:
    baseline = pd.read_parquet(LINEAR_ARTIFACTS / "oof_predictions.parquet")
    pd.testing.assert_frame_equal(baseline[["id", "label"]], train_meta, check_dtype=False)
    pd.testing.assert_frame_equal(baseline[list(folds.columns)], folds, check_dtype=False)
    scores = {"constant": baseline["p_repay_constant"].to_numpy(),
              "logistic_regression": baseline["p_repay_logistic"].to_numpy(), "lightgbm": lightgbm_oof}
    masks = {"overall": np.ones(len(folds), dtype=bool),
             "feature_492_upper_tail": folds["stress_feature_492_upper_tail"].to_numpy(dtype=bool),
             "feature_1039_observed": folds["stress_feature_1039_observed"].to_numpy(dtype=bool)}
    labels = train_meta["label"].to_numpy()
    return pd.DataFrame([{"model": model, "group": group, **metrics(labels[mask], p[mask])}
                         for model, p in scores.items() for group, mask in masks.items()])


def write_report(comparison: pd.DataFrame, fold_frame: pd.DataFrame,
                 importance: pd.DataFrame, config: dict) -> None:
    overview = comparison[["model", "group", "rows", "default_rate", "repay_roc_auc",
                           "default_pr_auc_ap", "log_loss", "brier_score"]].copy()
    overview["default_rate"] = overview["default_rate"].map(lambda x: f"{x:.4%}")
    for name in ["repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score"]:
        overview[name] = overview[name].map(lambda x: f"{x:.6f}")
    per_fold = fold_frame[["fold", "inner_fit_rows", "early_stop_rows", "rows", "selected_rounds",
                           "refit_rounds", "selection_hit_round_cap", "repay_roc_auc", "default_pr_auc_ap"]].copy()
    for name in ["repay_roc_auc", "default_pr_auc_ap"]:
        per_fold[name] = per_fold[name].map(lambda x: f"{x:.6f}")
    overall = comparison.query("group == 'overall'").set_index("model")
    top = importance.head(15)[["feature", "mean_gain_share", "mean_split_count"]].copy()
    top["mean_gain_share"] = top["mean_gain_share"].map(lambda x: f"{x:.2%}")
    top["mean_split_count"] = top["mean_split_count"].map(lambda x: f"{x:.1f}")
    lgbm = overall.loc["lightgbm"]
    logistic = overall.loc["logistic_regression"]
    audit_note = ""
    figure_note = ""
    inner_plot = REPORT / "figures/inner_learning_curves.png"
    metric_plot = REPORT / "figures/oof_metric_curves.png"
    if inner_plot.exists() and metric_plot.exists():
        figure_note = f"""## 学习用图

下面的曲线只使用内部早停集。圆点标出按 min_delta/patience 规则选出的树数，
不是外层最终评价；达到 600 轮上限的折仍可能受预算限制。

![内部早停曲线]({inner_plot.as_posix()})

下面使用全量 OOF 计算 ROC 和 AP，再对曲线渲染顶点抽稀。
PR 图使用阶梯线，避免把常数预测的人工端点连成有预测能力的斜线。

![折外指标对比]({metric_plot.as_posix()})

"""
    audit_path = REPORT / "artifact_verification.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "passed" and audit.get("training_signature") == config["training_signature"]:
            audit_note = "产物核查已通过：内部划分无交叉、外层折表未变、模型重载复现预测、指标与提交对齐均通过。\n"
    report = f"""# 第三步：LightGBM 基线

全量 **{config['train_rows']:,}** 条训练记录、**1,105** 个原始匿名特征，
沿用第二步五折。`label=1` 为还款；所有预测均为还款概率。
整体 OOF 还款 AUC **{lgbm['repay_roc_auc']:.6f}**，违约 AP **{lgbm['default_pr_auc_ap']:.6f}**，
相对逻辑回归 AUC 差为 **{lgbm['repay_roc_auc'] - logistic['repay_roc_auc']:+.6f}**。
log loss 为 **{lgbm['log_loss']:.6f}**，Brier 为 **{lgbm['brier_score']:.6f}**。
五折 AUC 均值 **{fold_frame['repay_roc_auc'].mean():.6f}**，标准差（ddof=1）
**{fold_frame['repay_roc_auc'].std(ddof=1):.6f}**；这不是官方测试成绩或置信区间。

## 验证协议与早停

每次留出一折作外层最终评价。从其余四折中按标签分层取 **{config['early_stop_fraction']:.0%}**
作为内部早停集，随机种子 `42 + fold`。第一阶段只在内部拟合集建立分箱、训练树，
早停集通过 `reference=inner_data` 使用相同分箱，监控 AUC 选择树数。
预先固定最多 **{config['max_rounds']}** 轮，连续 **{config['patience']}** 轮没有达到
至少 **{config['min_delta']}** 的有效 AUC 改善则早停。选好树数后，释放第一阶段模型，
在完整拟合四折重新建立分箱、从头训练同样数量的树；不再早停。
最后才预测外层验证折。外层验证数据和标签、测试数据均不参与分箱和树数选择。

达到树数预算上限的折数为 **{int(fold_frame['selection_hit_round_cap'].sum())}/5**。
达到上限只表示本轮计算预算用完，不能宣称已经找到全局最佳树数。
内部早停 AUC 属于用于选择的分数，不替代外层 OOF 指标。

{markdown_table(per_fold)}

## 与常数、逻辑回归的相同折外比较

从上一轮保存的 OOF 重新计算参照指标，并核对 id、标签、折号和压力标记完全一致。
违约 AP 定义为 `average_precision_score(1-label, 1-p_repay)`。
ROC-AUC 和 AP 越高越好；log loss 和 Brier 越低越好。

{markdown_table(overview)}

两个压力组由第二步提前确定，本次只对子集 OOF 概率计算指标，不重新选阈值或训练。
各组违约率不同，AP 需要结合基准率理解；AUC 也可能随样本组成变化。
随机折不能验证测试集超出训练取值范围的未知标签。

## 固定模型配置

```python
params = {{
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
    "min_data_in_leaf": 200, "lambda_l2": 1.0,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "max_bin": 63, "bin_construct_sample_cnt": 20000,
    "use_missing": True, "zero_as_missing": False,
    "deterministic": True, "force_col_wise": True,
}}
```

保留原始 NaN，由每个分裂学习缺失值方向；实际零值不当作缺失。
不做均值填补、标准化、人工缺失指示、异常值剪裁、特征筛选或类别重加权。
测试全空的 7 列仍保留，作为与前一基线一致的初始特征集合。
使用 CPU {config['threads']} 线程；`max_bin=63` 和 20,000 行分箱样本控制资源开销。
分箱抽样只估计分箱边界；模型阶段使用全部相应拟合行，行采样仅发生在每轮建树。
`IndexedSequence` 分批读取磁盘 float32 原始特征，并按接口要求将当前批转换为 float64。
LightGBM 仍需在内存保存分箱数据和直方图；不是零内存训练。

## 特征重要性

每折完整重训模型的分裂增益先归一化为 gain share，再取五折均值。
split count 为使用该特征分裂的次数。它们描述已训练模型，不是因果效应、系数方向，
也不是排列重要性；相关特征会分摊增益，高基数特征可能更容易被选中。
本轮没有根据该重要性删列或重新调参。

{markdown_table(top)}

## 运行产物与复现

- `artifacts/lightgbm_baseline/models/fold_*.txt` 与 JSON：用于 OOF/测试预测的完整四折重训模型及参数。
- `artifacts/lightgbm_baseline/inner_splits/fold_*.npz`：内部拟合、早停、外层验证的原始训练行位置。
- `artifacts/lightgbm_baseline/oof_predictions.parquet`：id、标签、固定折号、压力标记、OOF 还款概率。
- `artifacts/lightgbm_baseline/test_predictions.parquet`：测试 id、五折还款概率均值。
- `submissions/lightgbm_baseline.csv`：与官方模板 id 一对一对齐的本地提交，未上传 Kaggle。
- `reports/lightgbm_baseline/learning_curves/fold_*.json`：内部早停集逐轮 AUC。
- `comparison_metrics.csv`、`fold_metrics.csv`、`feature_importance.csv`、`summary.json`、`run_config.json`：结果与配置。

训练全部阶段耗时 **{config['elapsed_seconds']:.1f} 秒**。
记录输入文件来源、折表 SHA256、库版本和训练签名，复用上一轮原始特征缓存。
checkpoint 存储完整折结果，`--resume` 只在配置/来源一致时复用已完成折。
{audit_note}
```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe .\\scripts\\run_lightgbm_baseline.py
.\\.venv\\Scripts\\python.exe .\\scripts\\verify_lightgbm_baseline.py
.\\.venv\\Scripts\\python.exe .\\scripts\\plot_lightgbm_baseline.py
```

{figure_note}
阅读代码与面试准备见 `docs/lightgbm_baseline.md`。
"""
    (REPORT / "lightgbm_baseline_report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    for path in [REPORT, REPORT / "learning_curves", REPORT / "feature_importance",
                 ARTIFACTS / "models", ARTIFACTS / "inner_splits", ARTIFACTS / "fold_predictions"]:
        path.mkdir(parents=True, exist_ok=True)
    names = model_feature_columns(pq.read_schema(TRAIN).names, pq.read_schema(TEST).names)
    train_meta = pd.read_parquet(TRAIN, columns=["id", "label"])
    test_meta = pd.read_parquet(TEST, columns=["id"])
    folds = align_validation_folds(train_meta["id"], pd.read_parquet(FOLDS))
    if not train_meta["label"].isin([0, 1]).all() or set(folds["fold"].unique()) != set(range(5)):
        raise ValueError("Expected binary labels and all five saved folds")
    labels = train_meta["label"].to_numpy(dtype=np.int8)
    fold_id = folds["fold"].to_numpy()
    previous = json.loads((ROOT / "reports/linear_baselines/run_config.json").read_text(encoding="utf-8"))
    fold_sha = hashlib.sha256(FOLDS.read_bytes()).hexdigest()
    if previous["status"] != "completed" or previous["fold_sha256"] != fold_sha:
        raise RuntimeError("The reference baseline must be completed on the same folds")
    if previous["train_source"] != source_signature(TRAIN) or previous["test_source"] != source_signature(TEST):
        raise RuntimeError("Input data changed since the reference baseline")
    config = {"protocol_version": 1, "train_rows": len(labels), "test_rows": len(test_meta),
              "feature_names": names, "feature_count": len(names), "train_source": source_signature(TRAIN),
              "test_source": source_signature(TEST), "fold_sha256": fold_sha, "seed": SEED, "n_splits": 5,
              "reference_oof_source": source_signature(LINEAR_ARTIFACTS / "oof_predictions.parquet"),
              "early_stop_fraction": args.early_stop_fraction, "max_rounds": args.max_rounds,
              "patience": args.patience, "min_delta": args.min_delta, "batch_rows": args.batch_rows,
              "threads": args.threads, "params_fold_0": model_params(0, args.threads),
              "lightgbm_version": lgb.__version__, "numpy_version": np.__version__,
              "preprocessing": "native_missing_no_imputation_no_scaling",
              "test_prediction": "five_outer_refit_model_probability_mean",
              "initial_available_memory_gb": psutil.virtual_memory().available / 1e9}
    # Available memory is observational, not a model configuration change.
    signing = {k: v for k, v in config.items() if k != "initial_available_memory_gb"}
    signature = hashlib.sha256(json.dumps(signing, sort_keys=True).encode()).hexdigest()
    if args.resume and (REPORT / "run_config.json").exists():
        old = json.loads((REPORT / "run_config.json").read_text(encoding="utf-8"))
        if old["training_signature"] != signature:
            raise RuntimeError("Cannot resume: training configuration or input sources changed")
    config.update(status="running", training_signature=signature)
    json_write(REPORT / "run_config.json", config)
    values = raw_feature_cache(TRAIN, names, args.batch_rows)
    test_values = raw_feature_cache(TEST, names, args.batch_rows)
    oof = np.full(len(labels), np.nan)
    coverage = np.zeros(len(labels), dtype=np.int8)
    test_score = np.zeros(len(test_meta))
    records = []
    resumed = []
    for fold in range(5):
        checkpoint = ARTIFACTS / f"fold_predictions/fold_{fold}.npz"
        metadata = ARTIFACTS / f"models/fold_{fold}.json"
        if args.resume and checkpoint.exists() and metadata.exists():
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if saved.get("training_signature") != signature:
                raise RuntimeError("Fold checkpoint has a different training signature")
            model_path = ARTIFACTS / f"models/fold_{fold}.txt"
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != saved["model_sha256"]:
                raise RuntimeError("Saved model checksum mismatch")
            record = saved["record"]
            resumed.append(fold)
            print(f"Reusing completed fold {fold + 1}/5", flush=True)
        else:
            record = run_fold(values, test_values, labels, fold_id, names, fold, args)
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            saved["training_signature"] = signature
            json_write(metadata, saved)
        with np.load(checkpoint) as stored:
            valid = stored["outer_valid_index"]
            np.testing.assert_array_equal(valid, np.flatnonzero(fold_id == fold))
            if len(stored["p_repay_test"]) != len(test_meta):
                raise RuntimeError("Test prediction row count mismatch")
            oof[valid] = stored["p_repay_valid"]
            coverage[valid] += 1
            test_score += stored["p_repay_test"] / 5
        records.append(record)
        pd.DataFrame(records).to_csv(REPORT / "fold_metrics.csv", index=False)
    if not (coverage == 1).all() or not np.isfinite(oof).all():
        raise RuntimeError("OOF coverage is incomplete or duplicated")
    comparison = comparison_metrics(train_meta, folds, oof)
    comparison.to_csv(REPORT / "comparison_metrics.csv", index=False)
    frame = folds.copy()
    frame["label"] = labels
    frame["p_repay"] = oof
    frame.to_parquet(ARTIFACTS / "oof_predictions.parquet", index=False)
    predictions = test_meta.copy()
    predictions["p_repay"] = test_score
    predictions.to_parquet(ARTIFACTS / "test_predictions.parquet", index=False)
    template = pd.read_parquet(ROOT / "data/processed/sample_submission.parquet")
    build_submission(predictions, template).to_csv(ROOT / "submissions/lightgbm_baseline.csv", index=False)
    all_importance = pd.concat([pd.read_csv(REPORT / f"feature_importance/fold_{fold}.csv") for fold in range(5)])
    importance = all_importance.groupby("feature", as_index=False).agg(
        mean_gain_share=("gain_share", "mean"), mean_split_count=("split_count", "mean"),
        gain_share_std=("gain_share", "std"),
    ).sort_values("mean_gain_share", ascending=False).reset_index(drop=True)
    importance.to_csv(REPORT / "feature_importance.csv", index=False)
    fold_frame = pd.DataFrame(records)
    json_write(REPORT / "summary.json", {"groups": comparison.to_dict(orient="records"),
               "lightgbm_fold_auc_mean": float(fold_frame["repay_roc_auc"].mean()),
               "lightgbm_fold_auc_std_ddof1": float(fold_frame["repay_roc_auc"].std(ddof=1)),
               "selected_rounds": fold_frame["selected_rounds"].tolist(),
               "oof_coverage_min": int(coverage.min()), "oof_coverage_max": int(coverage.max()),
               "submissions_uploaded": False})
    config.update(status="completed", elapsed_seconds=time.perf_counter() - start,
                  all_fold_stage_seconds=float(fold_frame["elapsed_seconds"].sum()), resumed_folds=resumed)
    json_write(REPORT / "run_config.json", config)
    write_report(comparison, fold_frame, importance, config)
    print(comparison.to_string(index=False), flush=True)
    print(f"Completed. Report: {REPORT / 'lightgbm_baseline_report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds", type=int, default=600)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--early-stop-fraction", type=float, default=0.15)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_rounds, args.patience, args.batch_rows, args.threads) < 1 or args.min_delta < 0:
        parser.error("Counts must be positive and min_delta nonnegative")
    if not 0 < args.early_stop_fraction < 1:
        parser.error("early-stop-fraction must be between zero and one")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
