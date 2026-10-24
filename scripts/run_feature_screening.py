"""Fold-local missing-pattern and feature-filter ablations at fixed tree counts.

All four experiments are declared before training. Reuse the baseline's tree
counts, parameters, folds and seeds; fit feature rules on the fitting four folds.
The test-all-null and drift candidates come from earlier unlabeled diagnostics.
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
from threadpoolctl import threadpool_limits

try:
    from . import run_lightgbm_baseline as base
    from .id_feature_contract import align_validation_folds, build_submission
except ImportError:
    import run_lightgbm_baseline as base
    from id_feature_contract import align_validation_folds, build_submission


ROOT = base.ROOT
REPORT = ROOT / "reports/feature_screening"
ARTIFACTS = ROOT / "artifacts/feature_screening"
VARIANTS = ("missing_patterns", "availability_filter", "drift_core_filter", "low_information_filter")
TEST_ALL_NULL = ("feature_364", "feature_614", "feature_648", "feature_660",
                 "feature_805", "feature_1009", "feature_1031")
DRIFT_CORE = ("feature_492", "feature_903", "feature_586", "feature_1039")
RULES = {"missing_label_gap": 0.10, "high_missing_rate": 0.90,
         "dominant_state_share": 0.995,
         "test_all_null_candidates": list(TEST_ALL_NULL), "drift_candidates": list(DRIFT_CORE)}
DISPLAY = {"baseline": "Raw LightGBM", "missing_patterns": "Missing patterns",
           "availability_filter": "Availability filter", "drift_core_filter": "Drift-core filter",
           "low_information_filter": "Low-information filter"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def column_statistics(raw: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict:
    """Exact statistics of selected rows; NaN is a separate state for dominance.

    Observed variance is descriptive only: anonymous columns have different units.
    A constant observed value with a variable missing mask is not a constant input.
    """
    selected = np.asarray(raw)[indices]
    y = labels[indices]
    if len(selected) == 0 or set(np.unique(y)) != {0, 1} or np.isinf(selected).any():
        raise ValueError("Expected nonempty finite-or-NaN data and both label classes")
    missing = np.isnan(selected)
    observed = selected[~missing]
    if len(observed):
        unique, counts = np.unique(observed, return_counts=True)
        modal_count = int(counts.max())
        variance = float(np.var(observed, dtype=np.float64))
        observed_unique = len(unique)
    else:
        modal_count, observed_unique, variance = 0, 0, None
    missing_count = int(missing.sum())
    default_missing = float(missing[y == 0].mean())
    repay_missing = float(missing[y == 1].mean())
    return {"fit_rows": len(selected), "missing_rate": missing_count / len(selected),
            "default_missing_rate": default_missing, "repay_missing_rate": repay_missing,
            "abs_missing_label_gap": abs(repay_missing - default_missing),
            "observed_unique": observed_unique, "observed_variance": variance,
            "observed_modal_share": modal_count / len(observed) if len(observed) else None,
            "dominant_state_share": max(modal_count, missing_count) / len(selected)}


def profile_all_folds(values, labels: np.ndarray, fold_id: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = {fold: np.flatnonzero(fold_id != fold) for fold in range(5)}
    records = []
    start = time.perf_counter()
    for column, name in enumerate(names):
        # One raw column is copied, never the complete 3 GB training matrix.
        raw = np.array(values[:, column], dtype=np.float32, copy=True)
        for fold in range(5):
            records.append({"fold": fold, "feature": name,
                            **column_statistics(raw, labels, rows[fold])})
        if (column + 1) % 100 == 0 or column + 1 == len(names):
            print(f"Profiled {column + 1}/{len(names)} columns for all five fitting folds "
                  f"({time.perf_counter() - start:.0f}s)", flush=True)
    return pd.DataFrame(records)


def fit_plan(profile: pd.DataFrame, names: list[str], variant: str) -> dict:
    if variant not in VARIANTS or set(profile["feature"]) != set(names) or not profile["feature"].is_unique:
        raise ValueError("Invalid variant or feature profile")
    indexed = profile.set_index("feature").loc[names]
    reasons = {}
    indicators = []
    if variant == "missing_patterns":
        indicators = indexed.index[indexed["abs_missing_label_gap"] >= RULES["missing_label_gap"]].tolist()
    elif variant == "availability_filter":
        reasons = {name: "test_all_null_predeclared" for name in names if name in TEST_ALL_NULL}
        for name in indexed.index[indexed["missing_rate"] >= RULES["high_missing_rate"]]:
            reasons[name] = "fit_missing_rate_at_least_90_percent"
    elif variant == "drift_core_filter":
        reasons = {name: "drift_candidate_predeclared" for name in names if name in DRIFT_CORE}
    elif variant == "low_information_filter":
        for name in indexed.index[indexed["dominant_state_share"] >= RULES["dominant_state_share"]]:
            reasons[name] = "dominant_observed_or_missing_state_at_least_99_5_percent"
    kept = [name for name in names if name not in reasons]
    if not kept:
        raise ValueError("Feature filtering removed all original columns")
    add_count = variant == "missing_patterns"
    output_names = kept + [f"missing__{name}" for name in indicators]
    if add_count:
        output_names.append("row_missing_count")
    return {"variant": variant, "original_features": names, "kept_features": kept,
            "removed": reasons, "missing_indicators": indicators,
            "add_row_missing_count": add_count, "output_features": output_names,
            "fit_rows": int(indexed["fit_rows"].iloc[0]),
            "rules": RULES, "identity": output_names == names}


class FeatureView:
    """Apply an already-fitted schema to a requested raw row or batch only."""

    ndim = 2

    def __init__(self, values, plan: dict):
        self.values, self.plan = values, plan
        names = plan["original_features"]
        if values.shape[1] != len(names) or len(set(names)) != len(names):
            raise ValueError("Raw feature schema differs from the fitted plan")
        self.kept = np.array([names.index(name) for name in plan["kept_features"]], dtype=np.int64)
        self.indicators = np.array([names.index(name) for name in plan["missing_indicators"]], dtype=np.int64)
        self.shape = (len(values), len(plan["output_features"]))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        raw = np.asarray(self.values[item], dtype=np.float32)
        single = raw.ndim == 1
        if single:
            raw = raw[None, :]
        if self.plan["identity"]:
            return raw[0] if single else raw
        pieces = [raw[:, self.kept]]
        if len(self.indicators):
            pieces.append(np.isnan(raw[:, self.indicators]).astype(np.float32))
        if self.plan["add_row_missing_count"]:
            pieces.append(np.isnan(raw).sum(axis=1, keepdims=True).astype(np.float32))
        result = np.concatenate(pieces, axis=1)
        return result[0] if single else result


def group_metrics(labels: np.ndarray, folds: pd.DataFrame, predictions: np.ndarray) -> list[dict]:
    masks = {"overall": np.ones(len(labels), dtype=bool),
             "feature_492_upper_tail": folds["stress_feature_492_upper_tail"].to_numpy(dtype=bool),
             "feature_1039_observed": folds["stress_feature_1039_observed"].to_numpy(dtype=bool)}
    return [{"group": group, **base.metrics(labels[mask], predictions[mask])}
            for group, mask in masks.items()]


def load_context(batch_rows: int):
    previous = json.loads((base.REPORT / "run_config.json").read_text(encoding="utf-8"))
    if previous["status"] != "completed" or previous["fold_sha256"] != sha256(base.FOLDS):
        raise RuntimeError("Baseline is not complete on the current folds")
    if previous["train_source"] != base.source_signature(base.TRAIN) or previous["test_source"] != base.source_signature(base.TEST):
        raise RuntimeError("Input data changed since the baseline")
    if previous["params_fold_0"] != base.model_params(0, previous["threads"]):
        raise RuntimeError("Baseline parameters changed")
    meta = pd.read_parquet(base.TRAIN, columns=["id", "label"])
    test_meta = pd.read_parquet(base.TEST, columns=["id"])
    folds = align_validation_folds(meta["id"], pd.read_parquet(base.FOLDS))
    baseline_oof = pd.read_parquet(base.ARTIFACTS / "oof_predictions.parquet")
    pd.testing.assert_frame_equal(baseline_oof[["id", "label"]], meta, check_dtype=False)
    pd.testing.assert_frame_equal(baseline_oof[list(folds.columns)], folds, check_dtype=False)
    labels, fold_id = meta["label"].to_numpy(), folds["fold"].to_numpy()
    names = previous["feature_names"]
    values = base.raw_feature_cache(base.TRAIN, names, batch_rows)
    test_values = base.raw_feature_cache(base.TEST, names, batch_rows)
    # Audit the predeclared test-availability list without selecting with labels.
    actual_null = [name for j, name in enumerate(names) if np.isnan(test_values[:, j]).all()]
    if set(actual_null) != set(TEST_ALL_NULL):
        raise RuntimeError("Predeclared all-null list no longer matches the test data")
    return previous, meta, test_meta, folds, labels, fold_id, names, values, test_values, baseline_oof


def run_variant_fold(values, test_values, labels, fold_id, names, fold, plan, args, signature) -> dict:
    variant = plan["variant"]
    output = ARTIFACTS / variant
    fitting = np.flatnonzero(fold_id != fold)
    valid = np.flatnonzero(fold_id == fold)
    baseline_meta = json.loads((base.ARTIFACTS / f"models/fold_{fold}.json").read_text(encoding="utf-8"))
    rounds = baseline_meta["record"]["refit_rounds"]
    params = base.model_params(fold, args.threads)
    start = time.perf_counter()
    if plan["identity"]:
        model_path = base.ARTIFACTS / f"models/fold_{fold}.txt"
        with np.load(base.ARTIFACTS / f"fold_predictions/fold_{fold}.npz") as stored:
            np.testing.assert_array_equal(stored["outer_valid_index"], valid)
            p_valid, p_test = stored["p_repay_valid"], stored["p_repay_test"]
        origin = "reused_identical_baseline"
        print(f"{variant} fold {fold}: identity schema; reuse baseline predictions", flush=True)
    else:
        view, test_view = FeatureView(values, plan), FeatureView(test_values, plan)
        print(f"{variant} fold {fold}: fit={len(fitting):,}, original={len(names)}, "
              f"removed={len(plan['removed'])}, added={len(plan['output_features']) - len(plan['kept_features'])}, "
              f"trees={rounds}", flush=True)
        dataset = base.make_dataset(view, labels, fitting, plan["output_features"], params, args.batch_rows)
        # Tree count fixed before this experiment. No outer validation passed to train.
        model = lgb.train(params, dataset, num_boost_round=rounds,
                          callbacks=[base.log_refit(f"{variant} fold {fold}")])
        model_path = output / f"models/fold_{fold}.txt"
        model.save_model(str(model_path))
        del dataset
        gc.collect()
        p_valid = base.predict_rows(model, view, args.batch_rows, valid)
        p_test = base.predict_rows(model, test_view, args.batch_rows)
        gain = model.feature_importance(importance_type="gain")
        pd.DataFrame({"feature": plan["output_features"], "gain": gain,
                      "gain_share": gain / gain.sum() if gain.sum() else np.zeros_like(gain)}).to_csv(
                          REPORT / f"importance/{variant}_fold_{fold}.csv", index=False)
        del model
        origin = "trained_fixed_baseline_tree_count"
    record = {"variant": variant, "fold": fold, "fit_rows": len(fitting), "trees": rounds,
              "removed_features": len(plan["removed"]),
              "added_features": len(plan["output_features"]) - len(plan["kept_features"]),
              "output_features": len(plan["output_features"]), "origin": origin,
              **base.metrics(labels[valid], p_valid), "elapsed_seconds": time.perf_counter() - start}
    metadata = {"plan": plan, "record": record, "params": params, "positive_class": 1,
                "training_signature": signature, "model_path": str(model_path), "model_sha256": sha256(model_path)}
    base.json_write(output / f"models/fold_{fold}.json", metadata)
    np.savez_compressed(output / f"fold_predictions/fold_{fold}.npz", outer_valid_index=valid,
                        p_repay_valid=p_valid, p_repay_test=p_test)
    print(f"Finished {variant} fold {fold}: AUC={record['repay_roc_auc']:.6f}, "
          f"default AP={record['default_pr_auc_ap']:.6f}", flush=True)
    gc.collect()
    return record


def write_report(comparison: pd.DataFrame, fold_metrics: pd.DataFrame, plans: list[dict], config: dict):
    overall = comparison.query("group == 'overall'").copy()
    reference = overall.loc[overall.variant == "baseline"].iloc[0]
    overall["auc_delta"] = overall.repay_roc_auc - reference.repay_roc_auc
    overall = overall[["variant", "repay_roc_auc", "auc_delta", "default_pr_auc_ap", "log_loss", "brier_score"]]
    for column in overall.columns[1:]:
        overall[column] = overall[column].map(lambda value: f"{value:.6f}")
    groups = comparison[["variant", "group", "rows", "default_rate", "repay_roc_auc", "default_pr_auc_ap"]].copy()
    for column in ["default_rate", "repay_roc_auc", "default_pr_auc_ap"]:
        groups[column] = groups[column].map(lambda value: f"{value:.6f}")
    counts = pd.DataFrame([{"variant": plan["variant"], "fold": plan["fold"],
                            "removed": len(plan["removed"]), "indicators": len(plan["missing_indicators"]),
                            "row_count_added": plan["add_row_missing_count"],
                            "output_features": len(plan["output_features"]), "identity": plan["identity"]}
                           for plan in plans])
    best = comparison.query("group == 'overall'").sort_values("repay_roc_auc", ascending=False, kind="stable").iloc[0]
    audit = ""
    audit_path = REPORT / "artifact_verification.json"
    if audit_path.exists():
        saved = json.loads(audit_path.read_text(encoding="utf-8"))
        if saved.get("status") == "passed" and saved.get("training_signature") == config["training_signature"]:
            audit = "产物核查通过：模型重载复现全部预测、重新拟合规则一致、OOF 每行覆盖一次、指标及提交对齐。"
    figures = ""
    figure_path = REPORT / "figures/ablation_comparison.png"
    if figure_path.exists():
        figures = f"![特征处理消融对比]({figure_path.as_posix()})\n"
    text = f"""# 缺失模式、漂移和低方差特征筛选

使用完整 {config['train_rows']:,} 条训练记录、原始 1,105 特征，沿用固定分层五折。
`label=1` 是还款，所有模型预测还款概率，违约 AP 使用 `1-label` 与 `1-p_repay`。

## 实验设计

本轮四个方案在训练前声明，单独与原始 LightGBM 比较，不把多个变更堆叠后归因。
各折固定上一轮内部早停选定并完整重训的树数：{config['fixed_rounds']}。
固定原有模型参数和随机种子，不在本轮重新选树数或调参。当前折的规则仅由其余四折拟合；
外层验证不用于规则学习、分箱、叶子拟合或树数选择。测试集的预先声明列名单来自以前的
无标签诊断，是利用测试协变量的针对性消融；没有用测试值学习填补、阈值或还款模型。

- `missing_patterns`：保留原始值和 NaN；在拟合四折内计算还款/违约缺失率差，绝对差至少
  0.10 的列增加二值缺失标记，另加原始 1,105 列的逐行缺失数量。
- `availability_filter`：删除预先确定的测试全空 7 列，再删除拟合四折缺失率至少 90% 的列。
  这是可用性组合方案，不能单独归因于其中某一种删除。
- `drift_core_filter`：删除预先声明的 `feature_492/903/586/1039`。前三列为此前主要来源
  分类与数值漂移候选，1039 为缺失率剧变候选。不是删除所有 541 个高漂移标记列。
- `low_information_filter`：检查精确众数/缺失状态占比，至少 99.5% 的拟合行处于同一状态
  才删除。NaN 单独算一个状态。非缺失值恒定但缺失模式变化的列仍可能保留。
  原始非缺失方差只报告，不按统一阈值删除匿名且尺度不同的特征。无列可删时复用原模型。

## 整体 OOF

{base.markdown_table(overall)}

本轮按整体 OOF AUC 排名的本地开发最优是 **{best['variant']}**，AUC **{best['repay_roc_auc']:.6f}**。
这是当前验证数据上的方案比较与选择，不能把选中后的分数当作新的完全独立测试结果。
五折差异只作为配对描述，不当作五个独立实验做显著性保证。

## 压力组

{base.markdown_table(groups)}

压力组沿用既有标记，用同一批 OOF 子集计算，不单独训练或选阈值；各组可重叠、违约率不同。
随机划分仍不能验证测试中超出训练范围且没有标签的取值。删除漂移列也不保证消除整体漂移。

## 每折特征规则

{base.markdown_table(counts)}

每折规则可以不同，预测必须使用该折保存的 feature plan 和列顺序。
`FeatureView` 在请求的行/批次上应用规则，原始 NaN/零值保留，不复制完整转换矩阵。
LightGBM 的分箱 Dataset 和直方图仍在内存；特征采样、行采样与数据分批是不同操作。

## 产物与核查

配置和训练签名在 `run_config.json`；完整折内统计在 `fold_feature_statistics.csv`；
精确删列原因和新增列在 `feature_plans.json`；各折指标在 `fold_metrics.csv`，配对差在
`paired_fold_deltas.csv`。每方案保存模型/规则、OOF、五折测试概率均值与本地提交，均未上传。
缺失特征的重要性单独记录，但不是因果效应。无变化方案明确记录复用而非伪造新训练。
训练耗时 {config.get('elapsed_seconds', 0):.1f} 秒。{audit}

{figures}

学习说明见 `docs/feature_screening.md`。这些实验回答本地验证下的收益和代价，未建立时间验证，
也没有证明修复了训练/测试漂移。
"""
    (REPORT / "feature_screening_report.md").write_text(text, encoding="utf-8")


def run(args):
    start = time.perf_counter()
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "importance").mkdir(exist_ok=True)
    for variant in VARIANTS:
        for directory in ["models", "fold_predictions"]:
            (ARTIFACTS / variant / directory).mkdir(parents=True, exist_ok=True)
    context = load_context(args.batch_rows)
    previous, meta, test_meta, folds, labels, fold_id, names, values, test_values, baseline_oof = context
    baseline_metadata = [json.loads((base.ARTIFACTS / f"models/fold_{f}.json").read_text(encoding="utf-8")) for f in range(5)]
    config = {"protocol": "fixed_baseline_tree_counts_fold_local_rules_v1", "train_rows": len(labels),
              "test_rows": len(test_meta), "original_features": names, "rules": RULES, "variants": list(VARIANTS),
              "fixed_rounds": [item["record"]["refit_rounds"] for item in baseline_metadata],
              "baseline_model_checksums": [item["model_sha256"] for item in baseline_metadata],
              "baseline_training_signature": previous["training_signature"], "fold_sha256": sha256(base.FOLDS),
              "train_source": base.source_signature(base.TRAIN), "test_source": base.source_signature(base.TEST),
              "baseline_oof_sha256": sha256(base.ARTIFACTS / "oof_predictions.parquet"),
              "baseline_config_sha256": sha256(base.REPORT / "run_config.json"),
              "threads": args.threads, "batch_rows": args.batch_rows, "lightgbm_version": lgb.__version__,
              "params_fold_0": base.model_params(0, args.threads),
              "code_sha256": {path.name: sha256(path) for path in [Path(__file__), Path(base.__file__),
                              ROOT / "scripts/run_linear_baselines.py", ROOT / "scripts/id_feature_contract.py"]}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    old_path = REPORT / "run_config.json"
    if args.resume and old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if old["training_signature"] != signature:
            raise RuntimeError("Cannot resume with changed sources, code or configuration")
    if args.threads != previous["threads"]:
        raise ValueError("Use the same CPU thread count as the baseline")
    config.update(training_signature=signature, status="running")
    base.json_write(old_path, config)
    profile_path = REPORT / "fold_feature_statistics.csv"
    marker = REPORT / "profile_signature.json"
    if args.resume and profile_path.exists() and marker.exists() and json.loads(marker.read_text())["signature"] == signature:
        profile = pd.read_csv(profile_path, float_precision="round_trip")
    else:
        profile = profile_all_folds(values, labels, fold_id, names)
        profile.to_csv(profile_path, index=False)
        base.json_write(marker, {"signature": signature, "profile_sha256": sha256(profile_path)})
    if json.loads(marker.read_text())["profile_sha256"] != sha256(profile_path):
        raise RuntimeError("Profile checksum mismatch")
    plans = []
    for variant in VARIANTS:
        for fold in range(5):
            plan = fit_plan(profile.loc[profile.fold == fold], names, variant)
            plan["fold"] = fold
            plans.append(plan)
    base.json_write(REPORT / "feature_plans.json", plans)
    comparison = [{"variant": "baseline", **item} for item in group_metrics(labels, folds, baseline_oof.p_repay.to_numpy())]
    records = []
    for variant in VARIANTS:
        oof, coverage, test_score = np.full(len(labels), np.nan), np.zeros(len(labels), dtype=np.int8), np.zeros(len(test_meta))
        output = ARTIFACTS / variant
        for fold in range(5):
            checkpoint, metadata_path = output / f"fold_predictions/fold_{fold}.npz", output / f"models/fold_{fold}.json"
            plan = next(item for item in plans if item["variant"] == variant and item["fold"] == fold)
            if args.resume and checkpoint.exists() and metadata_path.exists():
                saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                if saved["training_signature"] != signature or saved["plan"] != plan or sha256(Path(saved["model_path"])) != saved["model_sha256"]:
                    raise RuntimeError("Checkpoint configuration or model mismatch")
                record = saved["record"]
                print(f"Reuse complete {variant} fold {fold}", flush=True)
            else:
                record = run_variant_fold(values, test_values, labels, fold_id, names, fold, plan, args, signature)
            with np.load(checkpoint) as stored:
                valid = stored["outer_valid_index"]
                np.testing.assert_array_equal(valid, np.flatnonzero(fold_id == fold))
                oof[valid] = stored["p_repay_valid"]
                coverage[valid] += 1
                test_score += stored["p_repay_test"] / 5
            records.append(record)
            pd.DataFrame(records).to_csv(REPORT / "fold_metrics.csv", index=False)
            config["completed_variant_folds"] = len(records)
            base.json_write(old_path, config)
        if not (coverage == 1).all() or not np.isfinite(oof).all() or not np.isfinite(test_score).all():
            raise RuntimeError("Incomplete prediction coverage")
        prediction = folds.copy()
        prediction["label"], prediction["p_repay"] = labels, oof
        prediction.to_parquet(output / "oof_predictions.parquet", index=False)
        test_prediction = test_meta.copy()
        test_prediction["p_repay"] = test_score
        test_prediction.to_parquet(output / "test_predictions.parquet", index=False)
        template = pd.read_parquet(ROOT / "data/processed/sample_submission.parquet")
        build_submission(test_prediction, template).to_csv(ROOT / f"submissions/feature_screening_{variant}.csv", index=False)
        comparison.extend({"variant": variant, **item} for item in group_metrics(labels, folds, oof))
        pd.DataFrame(comparison).to_csv(REPORT / "comparison_metrics.csv", index=False)
    frame = pd.DataFrame(records)
    baseline_folds = pd.read_csv(base.REPORT / "fold_metrics.csv").set_index("fold")
    paired = frame[["variant", "fold", "repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score"]].copy()
    for metric in ["repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score"]:
        paired[f"delta_{metric}"] = paired[metric] - paired["fold"].map(baseline_folds[metric])
    paired.to_csv(REPORT / "paired_fold_deltas.csv", index=False)
    table = pd.DataFrame(comparison)
    ranked = table.query("group == 'overall'").sort_values("repay_roc_auc", ascending=False, kind="stable")
    summary = {"groups": comparison, "best_development_variant_by_auc": ranked.iloc[0]["variant"],
               "best_development_auc": float(ranked.iloc[0]["repay_roc_auc"]), "submissions_uploaded": False,
               "no_effect_variants": [variant for variant in VARIANTS if all(plan["identity"] for plan in plans if plan["variant"] == variant)]}
    base.json_write(REPORT / "summary.json", summary)
    config.update(status="completed", elapsed_seconds=time.perf_counter() - start)
    base.json_write(old_path, config)
    write_report(table, frame, plans, config)
    print(table.query("group == 'overall'").to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args()
    if min(args.threads, args.batch_rows) < 1:
        parser.error("Threads and batch rows must be positive")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
