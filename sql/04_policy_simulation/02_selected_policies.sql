-- Python selects one policy per role on base/development results only.
-- A/B below name prospective policies; there is no randomized assignment.
CREATE TABLE selected_policy_definitions AS SELECT * FROM _selected_policies;
CREATE TABLE selected_policy_results AS
SELECT d.prospective_arm, r.*
FROM approval_economic_results r JOIN selected_policy_definitions d USING (policy_id);

-- Keep per-application outcomes to support the NEXT experimental-design stage.
-- Principal repayment is not income; only interest is revenue. All applications
-- incur application cost, including rejected rows. Test realized profits stay NULL.
CREATE VIEW selected_policy_application_outcomes AS
SELECT d.prospective_arm, d.policy_id, p.dataset, p.policy_cohort, p.id,
       p.model_version, p.model_role, p.label, p.p_repay,
       (d.decision_rule = 'approve_all' OR
          (d.decision_rule = 'score_threshold' AND p.p_repay >= d.threshold)) AS approved,
       CASE WHEN p.label IS NOT NULL THEN
           CASE WHEN d.decision_rule = 'approve_all' OR
                 (d.decision_rule = 'score_threshold' AND p.p_repay >= d.threshold)
                THEN p.label * e.principal * e.interest_rate_for_horizon
                  - (1 - p.label) * e.principal * e.loss_given_default
                  - e.principal * e.funding_rate_for_horizon - e.servicing_cost_per_approved
                ELSE 0 END - e.application_cost_per_application END AS historical_simulated_net_profit,
       CASE WHEN d.decision_rule = 'approve_all' OR
                 (d.decision_rule = 'score_threshold' AND p.p_repay >= d.threshold)
            THEN p.p_repay * e.principal * e.interest_rate_for_horizon
                - (1 - p.p_repay) * e.principal * e.loss_given_default
                - e.principal * e.funding_rate_for_horizon - e.servicing_cost_per_approved
            ELSE 0 END - e.application_cost_per_application AS raw_score_implied_net_profit
FROM simulation_predictions p JOIN selected_policy_definitions d USING (model_version, model_role)
CROSS JOIN economic_assumptions e WHERE e.scenario_id = 'base';
