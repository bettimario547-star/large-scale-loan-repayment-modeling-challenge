-- This is an exploratory threshold grid. No threshold is selected, no money
-- assumptions are introduced, and uncalibrated scores are not realized rates.
CREATE TABLE approval_threshold_grid AS SELECT * FROM _approval_thresholds;
CREATE TABLE approval_policy_results AS
WITH counts AS (
    SELECT dataset, model_version, model_role, g.threshold,
           count(*) AS total_rows, count(label) AS total_labeled_rows,
           count(*) FILTER (WHERE p_repay >= g.threshold) AS approved_rows,
           count(label) FILTER (WHERE p_repay >= g.threshold) AS labeled_approved_rows,
           sum(label) FILTER (WHERE p_repay >= g.threshold) AS approved_repayments,
           sum(1 - label) FILTER (WHERE p_repay >= g.threshold) AS approved_defaults,
           sum(1 - label) FILTER (WHERE p_repay < g.threshold) AS rejected_defaults,
           sum(1 - label) AS total_defaults, avg(1 - label) AS population_default_rate,
           avg(p_default) FILTER (WHERE p_repay >= g.threshold) AS mean_model_p_default_approved
    FROM model_predictions CROSS JOIN approval_threshold_grid g
    GROUP BY dataset, model_version, model_role, g.threshold
)
SELECT dataset, model_version, model_role, threshold, 'p_repay >= threshold' AS approval_rule,
       'exploratory_not_selected' AS policy_status, total_rows, total_labeled_rows,
       approved_rows, total_rows - approved_rows AS rejected_rows, labeled_approved_rows,
       approved_rows::DOUBLE / total_rows AS approval_rate,
       1 - approved_rows::DOUBLE / total_rows AS rejection_rate,
       CASE WHEN total_labeled_rows > 0 THEN coalesce(approved_defaults, 0) END AS approved_default_rows,
       CASE WHEN total_labeled_rows > 0 THEN coalesce(approved_repayments, 0) END AS approved_repaid_rows,
       CASE WHEN total_labeled_rows > 0 THEN coalesce(rejected_defaults, 0) END AS rejected_default_rows,
       coalesce(approved_defaults, 0)::DOUBLE / nullif(labeled_approved_rows, 0) AS observed_default_rate_approved,
       coalesce(approved_repayments, 0)::DOUBLE / nullif(labeled_approved_rows, 0) AS observed_repayment_rate_approved,
       population_default_rate, mean_model_p_default_approved,
       population_default_rate - coalesce(approved_defaults, 0)::DOUBLE
           / nullif(labeled_approved_rows, 0) AS observed_default_rate_reduction,
       coalesce(rejected_defaults, 0)::DOUBLE / nullif(total_defaults, 0) AS rejected_default_capture_share
FROM counts;

CREATE TABLE champion_challenger_comparison AS
SELECT a.dataset, a.threshold, 'same_probability_threshold_not_same_approval_rate' AS comparison_basis,
       a.model_version AS champion_version, b.model_version AS challenger_version,
       a.approval_rate AS champion_approval_rate, b.approval_rate AS challenger_approval_rate,
       b.approval_rate - a.approval_rate AS delta_approval_rate,
       a.observed_default_rate_approved AS champion_observed_default_rate,
       b.observed_default_rate_approved AS challenger_observed_default_rate,
       b.observed_default_rate_approved - a.observed_default_rate_approved AS delta_observed_default_rate,
       a.mean_model_p_default_approved AS champion_mean_model_p_default,
       b.mean_model_p_default_approved AS challenger_mean_model_p_default
FROM approval_policy_results a JOIN approval_policy_results b USING (dataset, threshold)
WHERE a.model_role = 'champion' AND b.model_role = 'challenger';
