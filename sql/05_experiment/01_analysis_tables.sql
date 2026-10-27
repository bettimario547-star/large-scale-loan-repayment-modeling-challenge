-- All labels and profits below belong to an OFFLINE simulation, not a live trial.
CREATE TABLE experiment_population AS SELECT * FROM _population;
CREATE TABLE experiment_assignments AS SELECT * FROM _assignments;
CREATE TABLE eligibility_summary AS
SELECT dataset, policy_cohort, count(*) AS applications,
       count(*) FILTER (WHERE eligible_proxy) AS eligible_proxy_applications,
       avg(eligible_proxy::INTEGER) AS eligible_proxy_fraction,
       count(*) FILTER (WHERE eligible_proxy AND decision_disagreement) AS eligible_disagreements,
       count(*) FILTER (WHERE eligible_proxy AND approved_a AND approved_b) AS both_approve,
       count(*) FILTER (WHERE eligible_proxy AND NOT approved_a AND NOT approved_b) AS both_reject,
       avg(approved_a::INTEGER) FILTER (WHERE eligible_proxy) AS approval_rate_a_within_proxy,
       avg(approved_b::INTEGER) FILTER (WHERE eligible_proxy) AS approval_rate_b_within_proxy,
       CASE WHEN dataset='train' THEN avg(potential_profit_b-potential_profit_a) FILTER (WHERE eligible_proxy) END AS paired_historical_profit_delta,
       TRUE AS simulation_only
FROM experiment_population GROUP BY dataset,policy_cohort;
CREATE TABLE experiment_metrics AS
SELECT experiment_id, arm, count(*) AS randomized_applications,
       sum(assigned_approved::INTEGER) AS approved_applications,
       avg(assigned_approved::INTEGER) AS approval_rate,
       sum(simulated_observed_default::INTEGER) AS simulated_default_events,
       avg(simulated_observed_default::INTEGER) AS simulated_default_incidence_itt,
       sum(simulated_observed_default::INTEGER)::DOUBLE/nullif(sum(assigned_approved::INTEGER),0) AS simulated_default_rate_approved,
       sum(simulated_observed_repaid::INTEGER)::DOUBLE/nullif(sum(assigned_approved::INTEGER),0) AS simulated_repayment_rate_approved,
       sum(simulated_observed_net_profit) AS simulated_net_profit,
       avg(simulated_observed_net_profit) AS simulated_profit_per_randomized_application_itt,
       var_samp(simulated_observed_net_profit) AS simulated_profit_variance,
       NULL::DOUBLE AS actual_dpd30_rate, TRUE AS simulation_only
FROM experiment_assignments GROUP BY experiment_id,arm;

-- Concentration by anonymous model-score strata; these are not demographics.
CREATE TABLE risk_score_concentration AS
WITH cells AS (
    SELECT arm,risk_stratum,count(*) AS randomized_applications,
           sum(assigned_approved::INTEGER) AS approved_applications
    FROM experiment_assignments GROUP BY arm,risk_stratum
)
SELECT *,approved_applications::DOUBLE/nullif(sum(approved_applications) OVER(PARTITION BY arm),0) AS share_of_arm_approved
FROM cells;
