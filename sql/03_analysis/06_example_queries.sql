-- Open data/interim/sql_business_analysis.duckdb after running the pipeline.
SELECT model_role, repay_roc_auc, default_pr_auc_ap, ks, log_loss, brier_score
FROM model_performance_summary ORDER BY model_role;

-- Highest-risk decile: relative risk, not a customer demographic segment.
SELECT model_role, row_count, observed_default_rate, default_lift, default_capture_share
FROM risk_band_summary WHERE dataset = 'train' AND risk_decile = 10;

-- Test labels are unavailable: this query returns score distributions only.
SELECT model_role, risk_decile, row_count, population_share, mean_model_p_default
FROM risk_band_summary WHERE dataset = 'test' ORDER BY model_role, risk_decile;

-- Thresholds are illustrative; they have not been selected for an experiment.
SELECT model_role, threshold, approval_rate, observed_default_rate_approved,
       mean_model_p_default_approved
FROM approval_policy_results WHERE dataset = 'train' AND threshold IN (0.8, 0.9, 0.95)
ORDER BY threshold, model_role;

SELECT * FROM champion_challenger_comparison
WHERE dataset = 'train' AND threshold = 0.9;
