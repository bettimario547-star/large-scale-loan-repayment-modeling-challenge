"""Constant and L2 logistic baselines on the saved five repayment folds.

The full-batch logistic objective matches sklearn's binary LogisticRegression
(C, no class weighting, unpenalized intercept). Disk-backed float32 matrices and
chunked loss/gradient evaluation avoid loading the wide dataset into RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from threadpoolctl import threadpool_limits

try:
    from .id_feature_contract import align_validation_folds, build_submission, model_feature_columns
except ImportError:
    from id_feature_contract import align_validation_folds, build_submission, model_feature_columns


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/processed/train.parquet"
TEST = ROOT / "data/processed/test.parquet"
FOLDS = ROOT / "data/interim/validation_folds.parquet"
REPORT = ROOT / "reports/linear_baselines"
ARTIFACTS = ROOT / "artifacts/linear_baselines"


def blocks(n_rows: int, batch_rows: int):
    for start in range(0, n_rows, batch_rows):
        yield slice(start, min(start + batch_rows, n_rows))


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def raw_feature_cache(path: Path, names: list[str], batch_rows: int) -> np.memmap:
    """Cache only explicit features; preserve and verify Parquet row order."""
    cache = ARTIFACTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    array_path = cache / f"{path.stem}_features.npy"
    manifest_path = cache / f"{path.stem}_features.json"
    parquet = pq.ParquetFile(path)
    shape = (parquet.metadata.num_rows, len(names))
    manifest = {"source": source_signature(path), "features": names, "shape": list(shape), "dtype": "float32"}
    if array_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) == manifest:
            cached = np.load(array_path, mmap_mode="r")
            if cached.shape == shape and cached.dtype == np.float32:
                print(f"Reusing {path.stem} feature cache: {shape}", flush=True)
                return cached
            raise RuntimeError("Cache shape or dtype is inconsistent")
    # Invalidate the completion marker before rewriting any cache.
    if manifest_path.exists():
        manifest_path.unlink()
    values = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=shape)
    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=names, use_threads=False):
        for column, name in enumerate(names):
            with np.errstate(over="raise", invalid="raise"):
                raw = batch.column(column).to_numpy(zero_copy_only=False).astype(np.float32)
            if np.isinf(raw).any():
                raise ValueError(f"Infinite or float32-overflowing values in {name}")
            values[offset:offset + len(batch), column] = raw
        offset += len(batch)
    if offset != shape[0]:
        raise RuntimeError("Incomplete feature cache")
    values.flush()
    del values
    json_write(manifest_path, manifest)
    print(f"Built {path.stem} feature cache: {shape}", flush=True)
    return np.load(array_path, mmap_mode="r")


def fit_preprocessor(values: np.ndarray, fit_mask: np.ndarray, batch_rows: int) -> dict:
    """Fold-local mean imputation, then population-variance standardization.

    Merge centered sums of squares for numerical stability. Mean-imputed cells
    have zero centered deviation; divide the observed M2 by *all* fitting rows.
    An entirely missing or constant fitting column has scale=1.
    """
    count = np.zeros(values.shape[1], dtype=np.int64)
    mean = np.zeros(values.shape[1], dtype=np.float64)
    m2 = np.zeros(values.shape[1], dtype=np.float64)
    fit_rows = int(fit_mask.sum())
    if fit_rows == 0:
        raise ValueError("No fitting rows")
    for section in blocks(len(values), batch_rows):
        chunk = np.array(values[section][fit_mask[section]], dtype=np.float64, copy=True)
        if len(chunk) == 0:
            continue
        if np.isinf(chunk).any():
            raise ValueError("Infinite features are not supported")
        observed = ~np.isnan(chunk)
        batch_count = observed.sum(axis=0)
        np.nan_to_num(chunk, copy=False, nan=0.0)
        batch_mean = np.divide(chunk.sum(axis=0), batch_count, out=np.zeros_like(mean), where=batch_count > 0)
        chunk -= batch_mean
        chunk[~observed] = 0.0
        batch_m2 = np.einsum("ij,ij->j", chunk, chunk)
        total = count + batch_count
        delta = batch_mean - mean
        ratio = np.divide(batch_count, total, out=np.zeros_like(mean), where=total > 0)
        m2 += batch_m2 + delta**2 * count * ratio
        mean += delta * ratio
        count = total
    scale = np.sqrt(np.maximum(m2 / fit_rows, 0))
    scale[scale < 1e-12] = 1.0
    return {"mean": mean, "scale": scale, "observed_count": count, "fit_rows": fit_rows}


def transform(values: np.ndarray, prep: dict) -> np.ndarray:
    # Calculate in float64 before storing compact float32 model inputs.
    chunk = np.array(values, dtype=np.float64, copy=True)
    chunk -= prep["mean"]
    chunk /= prep["scale"]
    np.nan_to_num(chunk, copy=False, nan=0.0)
    transformed = chunk.astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessing produced non-finite features")
    return transformed


def fitting_matrix(values: np.ndarray, fit_mask: np.ndarray, prep: dict, batch_rows: int) -> np.memmap:
    """Reuse one bounded disk file across folds; never materialize X[fit_index]."""
    path = ARTIFACTS / "cache/standardized_fit.npy"
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(int(fit_mask.sum()), values.shape[1]))
    offset = 0
    for section in blocks(len(values), batch_rows):
        raw = values[section][fit_mask[section]]
        chunk = transform(raw, prep)
        matrix[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    if offset != len(matrix):
        raise RuntimeError("Fitting matrix row count changed")
    matrix.flush()
    del matrix
    return np.load(path, mmap_mode="r")


def logistic_loss_gradient(theta: np.ndarray, values: np.ndarray, labels: np.ndarray,
                           C: float, batch_rows: int) -> tuple[float, np.ndarray]:
    """Mean binary log loss + ||coef||²/(2*C*n), intercept unpenalized."""
    if C <= 0 or len(labels) != len(values) or len(labels) == 0:
        raise ValueError("C must be positive and X/y lengths must match")
    gradient = np.zeros_like(theta, dtype=np.float64)
    loss = 0.0
    coef = theta[:-1].astype(values.dtype, copy=False)
    for section in blocks(len(values), batch_rows):
        chunk = values[section]
        target = labels[section]
        score = (chunk @ coef).astype(np.float64) + theta[-1]
        loss += float(np.sum(np.logaddexp(0.0, score) - target * score))
        residual = expit(score) - target
        gradient[:-1] += chunk.T @ residual.astype(values.dtype)
        gradient[-1] += residual.sum()
    n_rows = len(labels)
    loss = loss / n_rows + float(theta[:-1] @ theta[:-1]) / (2 * C * n_rows)
    gradient /= n_rows
    gradient[:-1] += theta[:-1] / (C * n_rows)
    return loss, gradient


def fit_logistic(values: np.ndarray, labels: np.ndarray, C: float, batch_rows: int,
                 max_iter: int, tolerance: float, progress: str = "") -> tuple[np.ndarray, dict]:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary repayment classes are required")
    initial = np.zeros(values.shape[1] + 1, dtype=np.float64)
    repay_rate = float(labels.mean())
    initial[-1] = np.log(repay_rate / (1 - repay_rate))
    state = {"iteration": 0}
    start = time.perf_counter()

    def callback(intermediate_result):
        state["iteration"] += 1
        if progress and state["iteration"] % 10 == 0:
            print(f"{progress} iteration {state['iteration']}: objective={intermediate_result.fun:.7f}, "
                  f"elapsed={time.perf_counter() - start:.0f}s", flush=True)

    result = minimize(logistic_loss_gradient, initial, args=(values, labels, C, batch_rows),
                      method="L-BFGS-B", jac=True, callback=callback,
                      options={"maxiter": max_iter, "gtol": tolerance, "ftol": 1e-9, "maxls": 30})
    diagnostics = {"converged": bool(result.success), "message": str(result.message),
                   "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                   "objective": float(result.fun), "max_abs_gradient": float(np.max(np.abs(result.jac))),
                   "elapsed_seconds": time.perf_counter() - start}
    if not np.isfinite(result.x).all():
        raise RuntimeError("Non-finite logistic model parameters")
    return result.x, diagnostics


def predict(values: np.ndarray, theta: np.ndarray, prep: dict, batch_rows: int,
            mask: np.ndarray | None = None) -> np.ndarray:
    scores = []
    for section in blocks(len(values), batch_rows):
        raw = values[section] if mask is None else values[section][mask[section]]
        if len(raw):
            chunk = transform(raw, prep)
            scores.append(expit(chunk @ theta[:-1].astype(np.float32) + theta[-1]))
    return np.concatenate(scores).astype(np.float64)


def constant_predictions(fit_labels: np.ndarray, n_valid: int) -> np.ndarray:
    if len(fit_labels) == 0 or not np.isin(fit_labels, [0, 1]).all():
        raise ValueError("Constant baseline needs binary fitting labels")
    return np.full(n_valid, float(fit_labels.mean()), dtype=np.float64)


def metrics(labels: np.ndarray, p_repay: np.ndarray) -> dict:
    if len(labels) != len(p_repay) or not np.isfinite(p_repay).all() or np.any((p_repay < 0) | (p_repay > 1)):
        raise ValueError("Invalid repayment probabilities")
    return {"rows": len(labels), "default_rows": int((labels == 0).sum()),
            "default_rate": float((labels == 0).mean()),
            "repay_roc_auc": float(roc_auc_score(labels, p_repay)),
            "default_pr_auc_ap": float(average_precision_score(1 - labels, 1 - p_repay)),
            "log_loss": float(log_loss(labels, p_repay, labels=[0, 1])),
            "brier_score": float(brier_score_loss(labels, p_repay)),
            "accuracy_at_0_5": float(accuracy_score(labels, p_repay >= 0.5))}


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(summary: pd.DataFrame, fold_metrics: pd.DataFrame, config: dict) -> None:
    overview = summary[["model", "group", "rows", "default_rate", "repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score"]].copy()
    overview["default_rate"] = overview["default_rate"].map(lambda value: f"{value:.4%}")
    for col in ["repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score"]:
        overview[col] = overview[col].map(lambda value: f"{value:.6f}")
    per_fold = fold_metrics[["model", "fold", "repay_roc_auc", "default_pr_auc_ap", "log_loss", "converged", "iterations"]].copy()
    for col in ["repay_roc_auc", "default_pr_auc_ap", "log_loss"]:
        per_fold[col] = per_fold[col].map(lambda value: f"{value:.6f}")
    logistic = summary.query("model == 'logistic_regression' and group == 'overall'").iloc[0]
    constant = summary.query("model == 'constant' and group == 'overall'").iloc[0]
    converged = bool(fold_metrics.query("model == 'logistic_regression'")["converged"].all())
    logistic_folds = fold_metrics.query("model == 'logistic_regression'")["repay_roc_auc"]
    stress = summary.query("model == 'logistic_regression'").set_index("group")
    verification_path = REPORT / "artifact_verification.json"
    verification_note = ""
    if verification_path.exists():
        verified = json.loads(verification_path.read_text(encoding="utf-8"))
        if verified.get("status") == "passed":
            verification_note = ("产物核查已通过：折表未变、OOF 对齐正确、指标重算一致、五折模型重载后"
                                 "复现测试概率、两份 CSV 与模板 id 和保存概率一致。\n")
    text = f"""# 第三步：常数预测与逻辑回归基线

## 本次结果

使用全量 **{config['train_rows']:,}** 条训练记录、全部 **1,105** 个匿名特征，
沿用第二步保存的五折；每条记录只有一次折外预测（OOF）。`1=还款，0=违约`。
逻辑回归整体折外还款 ROC-AUC 为 **{logistic['repay_roc_auc']:.6f}**，
违约 PR-AUC（Average Precision）为 **{logistic['default_pr_auc_ap']:.6f}**。
常数基线的每折 ROC-AUC 都是 0.5；逻辑回归提供了可比较的线性模型起点。
本次五折优化器全部报告收敛：**{converged}**。
逻辑回归五折 AUC 均值为 **{logistic_folds.mean():.6f}**，
标准差（ddof=1）为 **{logistic_folds.std(ddof=1):.6f}**。

## 整体和压力分组的折外指标

{markdown_table(overview)}

`overall` 是整体；`feature_492_upper_tail` 是第二步预先保存的高尾组；
`feature_1039_observed` 是其非缺失组。分组没有重新拟合模型或重新选阈值。
PR-AUC 在这里明确定义为 `average_precision_score(1-label, 1-p_repay)`，
并非 PR 曲线的梯形积分。分组违约率不同，AP 也受基准发生率影响，不能只比较 AP。
ROC-AUC、AP 越高越好；log loss、Brier 越低越好。

逻辑回归在高尾组的 AUC 为 **{stress.loc['feature_492_upper_tail', 'repay_roc_auc']:.6f}**，
在非缺失组为 **{stress.loc['feature_1039_observed', 'repay_roc_auc']:.6f}**，均低于整体。
两组仍优于各自的常数参照，但存在分组性能下降，后续模型应继续使用相同压力组比较。

## 每折结果与优化状态

{markdown_table(per_fold)}

五折 ROC-AUC 均值与标准差、优化器迭代次数、终止原因和梯度大小保存在
`summary.json`、`fold_metrics.csv` 和各折模型 JSON。均值/标准差是五折描述统计，
不把它当作独立样本的置信区间。

## 常数基线：先确认评价指标没有误导我们

每折仅用拟合四折的 `label.mean()`，为验证折所有行输出同一个还款概率。
五折常数略有不同，所以拼接后的整体 OOF AUC 可能略低于 0.5，
这是折间基准概率差造成的排序现象，不表示常数有预测能力。
全量训练还款率是 **{config['repay_rate']:.6%}**。
常数模型按 0.5 阈值全部预测还款，准确率仍有 **{constant['accuracy_at_0_5']:.6%}**；
这个准确率不能证明模型识别了违约。测试常数提交使用全量训练还款率。

## 逻辑回归：缺失填充、标准化和 L2 正则

每折用拟合四折的非缺失均值填补缺失，再使用填补后拟合数据的标准差缩放。
填补后的缺失值在标准化空间为 0。填补参数和缩放参数都不使用验证折或测试集。
本基线不加入缺失指示变量、不按标签筛特征、不剪裁异常值、不使用类别权重。
保留全部 1,105 列（包括测试全空的 7 列），作为明确的初始参照；
测试全空列只能被均值填充，后续可单独做缺失指示与删列消融。
`id`、`label`、`fold` 和压力标记均不进入 X。

模型输出 `sigmoid(X @ coef + intercept)`，含义是还款概率。
固定 `C={config['C']}`，最小化：

```text
mean(logaddexp(0, score) - label * score) + ||coef||² / (2 * C * n_fit)
```

截距不惩罚，目标与 scikit-learn 的二分类 L2 `LogisticRegression` 一致。
使用 SciPy L-BFGS-B 做全批量优化，分块累加损失和梯度；不是抽样训练或 SGD。
特征落盘为 float32，统计量、参数、损失累加为 float64；矩阵乘法存在 float32
舍入误差。小型测试与 sklearn 预测进行数值对照。
终止上限为 {config['max_iter']} 次，梯度容差为 {config['tolerance']}；
这些参数在评价前固定，不使用外层验证折早停或调参。

## 文件与复现

- `artifacts/linear_baselines/oof_predictions.parquet`：id、标签、固定折号、两模型折外概率和压力标记。
- `artifacts/linear_baselines/test_predictions.parquet`：测试 id、常数还款概率、五折逻辑回归概率均值。
- `artifacts/linear_baselines/models/fold_*.npz`：各折系数、截距、填补均值、缩放参数；JSON 记录特征顺序和优化状态。
- `submissions/constant_baseline.csv`、`submissions/logistic_regression_baseline.csv`：按官方模板 id 一对一对齐的本地提交文件。
- `reports/linear_baselines/summary.json`、`summary_metrics.csv`、`fold_metrics.csv`、`run_config.json`：完整结果和运行配置。
- `reports/linear_baselines/artifact_verification.json`：另运行产物核查脚本后生成的检查结果。

逻辑回归测试预测来自五折模型的均值，没有另做全量重训。文件已生成，未上传 Kaggle，
因此没有官方测试集分数。缓存使用约 5.7 GB 磁盘，避免把宽表整体放入内存。

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe .\\scripts\\run_linear_baselines.py
.\\.venv\\Scripts\\python.exe .\\scripts\\verify_linear_baselines.py
```

本次运行耗时 **{config['elapsed_seconds']:.1f} 秒**。训练与折表来源、折表 SHA256、
库版本和完整配置保存在 `run_config.json`。再次运行会重新训练，复用来源匹配的特征缓存。
{verification_note}

## 结论的边界

这是训练分布内的线性基线。第二步已发现训练/测试强漂移；尤其大量测试
`feature_492` 值超出训练取值范围，压力组依旧不能验证那部分未知标签。
训练内较高的 AUC 不能等同于 Kaggle 表现。
下一阶段可在相同折号上训练树模型，再独立检验缺失指示和漂移列消融的作用。
"""
    (REPORT / "linear_baselines_report.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    REPORT.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    names = model_feature_columns(pq.read_schema(TRAIN).names, pq.read_schema(TEST).names)
    train_meta = pd.read_parquet(TRAIN, columns=["id", "label"])
    test_meta = pd.read_parquet(TEST, columns=["id"])
    folds = align_validation_folds(train_meta["id"], pd.read_parquet(FOLDS))
    y = train_meta["label"].to_numpy(dtype=np.int8)
    if not train_meta["label"].isin([0, 1]).all() or set(folds["fold"].unique()) != set(range(5)):
        raise ValueError("Expected binary labels and all five saved folds")
    fold_id = folds["fold"].to_numpy()
    config = {"status": "running", "train_rows": len(y), "test_rows": len(test_meta),
              "repay_rate": float(y.mean()), "feature_count": len(names), "feature_names": names,
              "train_source": source_signature(TRAIN), "test_source": source_signature(TEST),
              "fold_sha256": hashlib.sha256(FOLDS.read_bytes()).hexdigest(), "fold_path": str(FOLDS),
              "n_splits": 5, "C": args.C, "max_iter": args.max_iter, "tolerance": args.tolerance,
              "batch_rows": args.batch_rows, "threads": args.threads,
              "solver": "scipy_L-BFGS-B_full_batch_chunked", "preprocessing": "fold_local_mean_impute_standard_scale",
              "class_weight": None, "missing_indicators": False, "test_model": "five_fold_probability_mean",
              "numpy_version": np.__version__, "scipy_version": scipy.__version__, "sklearn_version": sklearn.__version__}
    json_write(REPORT / "run_config.json", config)
    values = raw_feature_cache(TRAIN, names, args.batch_rows)
    test_values = raw_feature_cache(TEST, names, args.batch_rows)
    oof = {model: np.full(len(y), np.nan) for model in ["constant", "logistic_regression"]}
    coverage = np.zeros(len(y), dtype=np.int8)
    test_score = np.zeros(len(test_meta), dtype=np.float64)
    fold_records = []
    for fold in range(5):
        fit = fold_id != fold
        valid = fold_id == fold
        prefix = f"Logistic fold {fold + 1}/5"
        oof["constant"][valid] = constant_predictions(y[fit], int(valid.sum()))
        prep = fit_preprocessor(values, fit, args.batch_rows)
        matrix = fitting_matrix(values, fit, prep, args.batch_rows)
        print(f"{prefix}: fitting {int(fit.sum()):,} rows, {len(names)} features", flush=True)
        theta, diagnostics = fit_logistic(matrix, y[fit], args.C, args.batch_rows,
                                          args.max_iter, args.tolerance, prefix)
        del matrix  # Close the mapping before the next fold rewrites the work file (Windows).
        if not diagnostics["converged"]:
            json_write(ARTIFACTS / f"models/fold_{fold}_failed.json", diagnostics)
            raise RuntimeError(f"{prefix} did not converge: {diagnostics['message']}. "
                               "Increase --max-iter; no completed result or submission is claimed.")
        oof["logistic_regression"][valid] = predict(values, theta, prep, args.batch_rows, valid)
        test_score += predict(test_values, theta, prep, args.batch_rows) / 5
        coverage[valid] += 1
        np.savez_compressed(ARTIFACTS / f"models/fold_{fold}.npz", coef=theta[:-1], intercept=theta[-1], **prep)
        json_write(ARTIFACTS / f"models/fold_{fold}.json", {"fold": fold, "classes": [0, 1], "features": names,
                   "C": args.C, "preprocessing_fit_rows": int(fit.sum()), "optimizer": diagnostics})
        for model, scores in oof.items():
            record = {"model": model, "fold": fold, "fit_rows": int(fit.sum()), **metrics(y[valid], scores[valid]),
                      "converged": True, "iterations": 0}
            if model == "logistic_regression":
                record.update(diagnostics)
            fold_records.append(record)
        pd.DataFrame(fold_records).to_csv(REPORT / "fold_metrics.csv", index=False)
        current = fold_records[-1]
        print(f"{prefix}: AUC={current['repay_roc_auc']:.6f}, default AP={current['default_pr_auc_ap']:.6f}, "
              f"log loss={current['log_loss']:.6f}, converged={diagnostics['converged']}", flush=True)
    if not (coverage == 1).all() or any(not np.isfinite(score).all() for score in oof.values()):
        raise RuntimeError("Incomplete or duplicate OOF prediction coverage")
    group_masks = {"overall": np.ones(len(y), dtype=bool),
                   "feature_492_upper_tail": folds["stress_feature_492_upper_tail"].to_numpy(dtype=bool),
                   "feature_1039_observed": folds["stress_feature_1039_observed"].to_numpy(dtype=bool)}
    summary = pd.DataFrame([{"model": model, "group": group, **metrics(y[mask], score[mask])}
                            for model, score in oof.items() for group, mask in group_masks.items()])
    summary.to_csv(REPORT / "summary_metrics.csv", index=False)
    frame = folds.copy()
    frame["label"] = y
    frame["p_repay_constant"] = oof["constant"]
    frame["p_repay_logistic"] = oof["logistic_regression"]
    frame.to_parquet(ARTIFACTS / "oof_predictions.parquet", index=False)
    predictions = test_meta.copy()
    predictions["p_repay_constant"] = float(y.mean())
    predictions["p_repay_logistic"] = test_score
    predictions.to_parquet(ARTIFACTS / "test_predictions.parquet", index=False)
    template = pd.read_parquet(ROOT / "data/processed/sample_submission.parquet")
    submissions = ROOT / "submissions"
    submissions.mkdir(exist_ok=True)
    for model, col in [("constant", "p_repay_constant"), ("logistic_regression", "p_repay_logistic")]:
        matched = predictions[["id", col]].rename(columns={col: "p_repay"})
        build_submission(matched, template).to_csv(submissions / f"{model}_baseline.csv", index=False)
    fold_frame = pd.DataFrame(fold_records)
    aggregates = []
    for model, subset in fold_frame.groupby("model", sort=False):
        aggregates.append({"model": model, "repay_auc_mean": float(subset["repay_roc_auc"].mean()),
                           "repay_auc_std_ddof1": float(subset["repay_roc_auc"].std(ddof=1)),
                           "default_ap_mean": float(subset["default_pr_auc_ap"].mean())})
    json_write(REPORT / "summary.json", {"groups": summary.to_dict(orient="records"), "fold_aggregates": aggregates,
                                         "oof_coverage_min": int(coverage.min()), "oof_coverage_max": int(coverage.max()),
                                         "submissions_uploaded": False})
    config.update(status="completed", elapsed_seconds=time.perf_counter() - start)
    json_write(REPORT / "run_config.json", config)
    write_report(summary, fold_frame, config)
    print(summary.to_string(index=False), flush=True)
    print(f"Completed in {config['elapsed_seconds']:.1f}s. Report: {REPORT / 'linear_baselines_report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.C <= 0 or args.max_iter < 1 or args.tolerance <= 0 or args.batch_rows < 1 or args.threads < 1:
        parser.error("All numeric options must be positive")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
