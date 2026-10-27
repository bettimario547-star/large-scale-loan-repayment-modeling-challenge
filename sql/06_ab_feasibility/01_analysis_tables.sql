-- D-only randomization; all outcome columns are historical simulations.
CREATE TABLE experiment_population AS SELECT * FROM _population;
CREATE TABLE experiment_assignments AS SELECT * FROM _assignments;
CREATE TABLE eligibility_summary AS
SELECT dataset,policy_cohort,count(*) AS applications,
  count(*) FILTER(WHERE eligible_proxy) AS common_eligible,
  count(*) FILTER(WHERE trial_eligible) AS disagreements,
  count(*) FILTER(WHERE eligible_proxy AND NOT decision_disagreement) AS agreements,
  count(*) FILTER(WHERE eligible_proxy AND approved_a) AS approved_a,
  count(*) FILTER(WHERE eligible_proxy AND approved_b) AS approved_b,
  avg(trial_eligible::INTEGER) AS disagreement_fraction_of_all,
  count(*) FILTER(WHERE trial_eligible)::DOUBLE/nullif(count(*) FILTER(WHERE eligible_proxy),0) AS disagreement_fraction_of_eligible,
  TRUE AS simulation_only
FROM experiment_population GROUP BY dataset,policy_cohort;
CREATE TABLE experiment_metrics AS
SELECT experiment_id,arm,count(*) AS randomized_disagreements,
  sum(assigned_approved::INTEGER) AS approved,
  sum(simulated_observed_default::INTEGER) AS defaults,
  sum(simulated_observed_net_profit) AS simulated_profit,
  avg(simulated_observed_net_profit) AS unadjusted_disagreement_profit,
  NULL::DOUBLE AS actual_dpd30_rate,TRUE AS simulation_only
FROM experiment_assignments GROUP BY experiment_id,arm;
CREATE TABLE observed_stratum_metrics AS
SELECT analysis_stratum AS stratum,arm,count(*) AS n,
  avg(simulated_observed_net_profit) AS profit_mean,
  var_samp(simulated_observed_net_profit) AS profit_variance,
  avg(simulated_observed_default::INTEGER) AS bad_mean,
  var_samp(simulated_observed_default::INTEGER) AS bad_variance,
  sum(assigned_approved::INTEGER) AS approved,sum(simulated_observed_default::INTEGER) AS bad
FROM experiment_assignments GROUP BY analysis_stratum,arm;
-- Full-population weights come from pre-treatment covariates, not outcomes.
CREATE TABLE target_stratum_weights AS
WITH totals AS (SELECT count(*) AS n FROM experiment_population
  WHERE dataset='train' AND policy_cohort='policy_evaluation' AND eligible_proxy)
SELECT analysis_stratum AS stratum,count(*) AS applications,
  count(*)::DOUBLE/(SELECT n FROM totals) AS weight_of_full_eligible
FROM experiment_population
WHERE dataset='train' AND policy_cohort='policy_evaluation' AND trial_eligible
GROUP BY analysis_stratum;
CREATE TABLE sql_primary_estimate AS
WITH contrast AS (
  SELECT a.stratum,w.weight_of_full_eligible AS w,b.profit_mean-a.profit_mean AS profit_difference,
    a.profit_variance/a.n+b.profit_variance/b.n AS independent_variance,
    b.bad_mean-a.bad_mean AS bad_difference
  FROM observed_stratum_metrics a JOIN observed_stratum_metrics b USING(stratum)
  JOIN target_stratum_weights w USING(stratum) WHERE a.arm='A' AND b.arm='B'
)
SELECT sum(w*profit_difference) AS full_eligible_profit_delta,
  sqrt(sum(w*w*independent_variance)) AS independent_stratified_se,
  sum(w*bad_difference) AS full_eligible_default_delta,
  TRUE AS simulation_only FROM contrast;
