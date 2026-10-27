-- Query data/interim/ab_test_design.duckdb. This is simulated evidence only.
SELECT * FROM eligibility_summary ORDER BY dataset,policy_cohort;
SELECT arm,randomized_applications,approval_rate,simulated_default_incidence_itt,
       simulated_default_rate_approved,simulated_profit_per_randomized_application_itt
FROM experiment_metrics ORDER BY arm;
SELECT * FROM assignment_quality ORDER BY risk_stratum;
SELECT endpoint,fraction_b,effect_or_margin,n_total,achieved_normal_power
FROM power_analysis ORDER BY endpoint,fraction_b,effect_or_margin;
SELECT * FROM bootstrap_results ORDER BY method;
SELECT * FROM experiment_duration ORDER BY daily_all_applications_assumed,eligible_fraction_assumed;
