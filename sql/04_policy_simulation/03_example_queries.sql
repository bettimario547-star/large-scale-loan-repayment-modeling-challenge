-- Use data/interim/approval_policy_simulation.duckdb.
SELECT prospective_arm, model_role, threshold, target_approval_rate
FROM selected_policy_definitions ORDER BY prospective_arm;

SELECT prospective_arm, approval_rate, observed_default_rate_approved,
       historical_profit_per_application, raw_score_implied_profit_per_application
FROM selected_policy_results WHERE scenario_id = 'base' AND policy_cohort = 'policy_evaluation'
ORDER BY prospective_arm;

-- A common target budget need not produce identical evaluation approval rates.
SELECT target_approval_rate, exactly_equal_approved_count,
       delta_approval_rate, delta_default_rate, delta_profit_per_application
FROM matched_approval_comparison
WHERE scenario_id = 'base' AND policy_cohort = 'policy_evaluation'
ORDER BY target_approval_rate;

-- Sensitivity evaluates the SAME chosen policies, without reselecting them.
SELECT scenario_id, prospective_arm, historical_profit_per_application
FROM selected_policy_results WHERE policy_cohort = 'policy_evaluation'
ORDER BY scenario_id, prospective_arm;
