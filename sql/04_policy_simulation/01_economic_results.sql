-- Inputs are immutable predictions, a common policy split, and thresholds
-- fitted ONLY on policy-development scores. No test-based threshold refit.
CREATE TABLE economic_assumptions AS SELECT * FROM _economic_assumptions;
CREATE TABLE policy_candidates AS SELECT * FROM _policy_candidates;
CREATE TABLE policy_constraints AS SELECT * FROM _policy_constraints;
CREATE TABLE policy_cohorts AS SELECT * FROM read_parquet('__SPLIT_PARQUET__');
CREATE VIEW simulation_predictions AS
SELECT p.*, CASE WHEN p.dataset = 'test' THEN 'unlabeled_test' ELSE c.policy_cohort END AS policy_cohort
FROM read_parquet('__PREDICTIONS_PARQUET__') p
LEFT JOIN policy_cohorts c ON p.dataset = 'train' AND p.id = c.id;

CREATE TABLE approval_economic_results AS
WITH counts AS (
    SELECT p.dataset, p.policy_cohort, p.model_version, p.model_role,
           c.policy_id, c.target_approval_rate, c.threshold, c.decision_rule,
           c.development_approved_rows, c.development_total_rows,
           count(*) AS total_applications, count(p.label) AS labeled_applications,
           count(*) FILTER (WHERE c.decision_rule = 'approve_all'
                 OR (c.decision_rule = 'score_threshold' AND p.p_repay >= c.threshold)) AS approved_applications,
           CASE WHEN count(p.label) > 0 THEN coalesce(sum(p.label) FILTER (WHERE c.decision_rule = 'approve_all'
                 OR (c.decision_rule = 'score_threshold' AND p.p_repay >= c.threshold)), 0) END AS approved_repaid_rows,
           CASE WHEN count(p.label) > 0 THEN coalesce(sum(1 - p.label) FILTER (WHERE c.decision_rule = 'approve_all'
                 OR (c.decision_rule = 'score_threshold' AND p.p_repay >= c.threshold)), 0) END AS approved_default_rows,
           sum(p.p_repay) FILTER (WHERE c.decision_rule = 'approve_all'
                 OR (c.decision_rule = 'score_threshold' AND p.p_repay >= c.threshold)) AS approved_score_sum
    FROM simulation_predictions p JOIN policy_candidates c USING (model_version, model_role)
    GROUP BY p.dataset, p.policy_cohort, p.model_version, p.model_role,
             c.policy_id, c.target_approval_rate, c.threshold, c.decision_rule,
             c.development_approved_rows, c.development_total_rows
), components AS (
    SELECT c.*, e.*,
           approved_applications::DOUBLE / total_applications AS approval_rate,
           CASE WHEN labeled_applications > 0 THEN coalesce(approved_default_rows, 0)::DOUBLE
                  / nullif(approved_applications, 0) END AS observed_default_rate_approved,
           CASE WHEN labeled_applications > 0 THEN coalesce(approved_repaid_rows, 0)
                  * principal * interest_rate_for_horizon END AS historical_simulated_interest_income,
           CASE WHEN labeled_applications > 0 THEN coalesce(approved_default_rows, 0)
                  * principal * loss_given_default END AS historical_simulated_principal_loss,
           approved_applications * principal * funding_rate_for_horizon AS assumed_funding_cost,
           approved_applications * servicing_cost_per_approved AS assumed_servicing_cost,
           total_applications * application_cost_per_application AS assumed_application_cost,
           coalesce(approved_score_sum, 0) * principal * interest_rate_for_horizon
             - (approved_applications - coalesce(approved_score_sum, 0)) * principal * loss_given_default
             - approved_applications * (principal * funding_rate_for_horizon + servicing_cost_per_approved)
             - total_applications * application_cost_per_application AS raw_score_implied_net_profit
    FROM counts c CROSS JOIN economic_assumptions e
), profits AS (
    SELECT *, historical_simulated_interest_income - historical_simulated_principal_loss
          - assumed_funding_cost - assumed_servicing_cost - assumed_application_cost AS historical_simulated_net_profit
    FROM components
)
SELECT *, total_applications - approved_applications AS rejected_applications,
       historical_simulated_net_profit / total_applications AS historical_profit_per_application,
       historical_simulated_net_profit / nullif(approved_applications, 0) AS historical_profit_per_approved_loan,
       raw_score_implied_net_profit / total_applications AS raw_score_implied_profit_per_application,
       approved_score_sum / nullif(approved_applications, 0) AS mean_p_repay_approved,
       CASE WHEN policy_cohort = 'policy_development' AND scenario_id = 'base' THEN
           observed_default_rate_approved <= k.max_development_default_rate
           AND approval_rate >= k.min_development_approval_rate ELSE NULL END AS eligible_for_selection
FROM profits CROSS JOIN policy_constraints k;

CREATE TABLE matched_approval_comparison AS
SELECT a.scenario_id, a.dataset, a.policy_cohort, a.target_approval_rate,
       a.policy_id AS champion_policy_id, b.policy_id AS challenger_policy_id,
       a.approved_applications AS champion_approved_rows, b.approved_applications AS challenger_approved_rows,
       a.approved_applications = b.approved_applications AS exactly_equal_approved_count,
       a.approval_rate AS champion_approval_rate, b.approval_rate AS challenger_approval_rate,
       b.approval_rate - a.approval_rate AS delta_approval_rate,
       a.observed_default_rate_approved AS champion_default_rate,
       b.observed_default_rate_approved AS challenger_default_rate,
       b.observed_default_rate_approved - a.observed_default_rate_approved AS delta_default_rate,
       a.historical_profit_per_application AS champion_profit_per_application,
       b.historical_profit_per_application AS challenger_profit_per_application,
       b.historical_profit_per_application - a.historical_profit_per_application AS delta_profit_per_application
FROM approval_economic_results a JOIN approval_economic_results b
USING (scenario_id, dataset, policy_cohort, target_approval_rate)
WHERE a.model_role = 'champion' AND b.model_role = 'challenger';
