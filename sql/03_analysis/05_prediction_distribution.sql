CREATE TABLE prediction_score_summary AS
SELECT dataset, prediction_kind, model_version, model_role, count(*) AS row_count,
       count(label) AS labeled_rows, avg(p_repay) AS mean_p_repay,
       stddev_pop(p_repay) AS std_p_repay, min(p_repay) AS min_p_repay,
       quantile_cont(p_repay, 0.01) AS p01, quantile_cont(p_repay, 0.1) AS p10,
       quantile_cont(p_repay, 0.5) AS p50, quantile_cont(p_repay, 0.9) AS p90,
       quantile_cont(p_repay, 0.99) AS p99, max(p_repay) AS max_p_repay,
       avg(p_default) AS mean_model_p_default, avg(1 - label) AS observed_default_rate
FROM model_predictions GROUP BY dataset, prediction_kind, model_version, model_role;

-- Score PSI uses the SAME train-OOF boundaries in both datasets.
-- It can reflect both data drift and different ensemble counts (OOF/test).
CREATE TABLE prediction_drift_summary AS
SELECT a.model_version, a.model_role, 'train_oof_deciles' AS boundary_reference,
       sum((greatest(b.population_share, 1e-6) - greatest(a.population_share, 1e-6))
           * ln(greatest(b.population_share, 1e-6) / greatest(a.population_share, 1e-6))) AS score_psi,
       0.5 * sum(abs(b.population_share - a.population_share)) AS decile_total_variation,
       'score_distribution_diagnostic_not_label_drift' AS interpretation
FROM risk_band_summary a JOIN risk_band_summary b USING (model_version, risk_decile)
WHERE a.dataset = 'train' AND b.dataset = 'test' GROUP BY a.model_version, a.model_role;
