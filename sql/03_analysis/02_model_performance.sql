-- Group tied scores before cumulative sums. Splitting a tie by id would
-- change AP and KS and could manufacture ranking ability in a constant model.
CREATE TABLE model_performance_summary AS
WITH score_groups AS (
    SELECT model_version, model_role, p_default,
           count(*) AS group_rows,
           sum(1 - label) AS group_defaults,
           sum(label) AS group_repayments
    FROM model_predictions WHERE dataset = 'train'
    GROUP BY model_version, model_role, p_default
), cumulative AS (
    SELECT *,
           sum(group_rows) OVER w AS cumulative_rows,
           sum(group_defaults) OVER w AS cumulative_defaults,
           sum(group_repayments) OVER w AS cumulative_repayments,
           sum(group_defaults) OVER (PARTITION BY model_version) AS total_defaults,
           sum(group_repayments) OVER (PARTITION BY model_version) AS total_repayments
    FROM score_groups
    WINDOW w AS (PARTITION BY model_version ORDER BY p_default DESC
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
), ranking AS (
    SELECT model_version, model_role,
           sum(group_defaults * (total_repayments - cumulative_repayments
                                 + 0.5 * group_repayments))
               / nullif(max(total_defaults) * max(total_repayments), 0) AS repay_roc_auc,
           sum(group_defaults * (cumulative_defaults::DOUBLE / cumulative_rows))
               / nullif(max(total_defaults), 0) AS default_pr_auc_ap,
           max(abs(cumulative_defaults::DOUBLE / nullif(total_defaults, 0)
                   - cumulative_repayments::DOUBLE / nullif(total_repayments, 0))) AS ks
    FROM cumulative GROUP BY model_version, model_role
), probability_errors AS (
    SELECT model_version, count(*) AS row_count, sum(label) AS repaid_rows,
           sum(1 - label) AS default_rows, avg(1 - label) AS observed_default_rate,
           avg(-label * ln(greatest(1e-15, least(1 - 1e-15, p_repay)))
               - (1 - label) * ln(greatest(1e-15, least(1 - 1e-15, p_default)))) AS log_loss,
           avg(pow(label - p_repay, 2)) AS brier_score
    FROM model_predictions WHERE dataset = 'train' GROUP BY model_version
)
SELECT 'train' AS dataset, 'reused_development_oof' AS evaluation_scope, *
FROM ranking JOIN probability_errors USING (model_version);
