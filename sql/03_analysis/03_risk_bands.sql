-- Boundaries are descriptive train-OOF quantiles, not policy selection or
-- calibration. Test labels/values cannot determine these boundaries.
CREATE TABLE risk_band_boundaries AS
SELECT model_version, model_role, 'train_oof' AS boundary_reference,
       quantile_cont(p_default, 0.1) AS p10, quantile_cont(p_default, 0.2) AS p20,
       quantile_cont(p_default, 0.3) AS p30, quantile_cont(p_default, 0.4) AS p40,
       quantile_cont(p_default, 0.5) AS p50, quantile_cont(p_default, 0.6) AS p60,
       quantile_cont(p_default, 0.7) AS p70, quantile_cont(p_default, 0.8) AS p80,
       quantile_cont(p_default, 0.9) AS p90
FROM model_predictions WHERE dataset = 'train' GROUP BY model_version, model_role;

-- Equal scores stay together. Consequently tied data need not give exactly
-- ten equally populated bands; empty bands are retained in the summary.
CREATE VIEW risk_band_assignments AS
SELECT p.*,
       CASE WHEN p.p_default <= b.p10 THEN 1 WHEN p.p_default <= b.p20 THEN 2
            WHEN p.p_default <= b.p30 THEN 3 WHEN p.p_default <= b.p40 THEN 4
            WHEN p.p_default <= b.p50 THEN 5 WHEN p.p_default <= b.p60 THEN 6
            WHEN p.p_default <= b.p70 THEN 7 WHEN p.p_default <= b.p80 THEN 8
            WHEN p.p_default <= b.p90 THEN 9 ELSE 10 END AS risk_decile
FROM model_predictions p JOIN risk_band_boundaries b USING (model_version);

CREATE TABLE risk_band_summary AS
WITH totals AS (
    SELECT dataset, model_version, model_role, count(*) AS total_rows,
           count(label) AS total_labeled_rows, sum(1 - label) AS total_defaults,
           sum(label) AS total_repayments, avg(1 - label) AS population_default_rate,
           avg(label) AS population_repayment_rate
    FROM model_predictions GROUP BY dataset, model_version, model_role
), grouped AS (
    SELECT dataset, model_version, risk_decile, count(*) AS row_count,
           count(label) AS labeled_rows, sum(1 - label) AS default_rows,
           sum(label) AS repaid_rows, avg(1 - label) AS observed_default_rate,
           avg(label) AS observed_repayment_rate, min(p_default) AS min_p_default,
           max(p_default) AS max_p_default, avg(p_default) AS mean_model_p_default
    FROM risk_band_assignments GROUP BY dataset, model_version, risk_decile
), complete AS (
    SELECT t.dataset, t.model_version, t.model_role, d.risk_decile,
           coalesce(g.row_count, 0) AS row_count, coalesce(g.labeled_rows, 0) AS labeled_rows,
           CASE WHEN t.total_labeled_rows > 0 THEN coalesce(g.default_rows, 0) END AS default_rows,
           CASE WHEN t.total_labeled_rows > 0 THEN coalesce(g.repaid_rows, 0) END AS repaid_rows,
           coalesce(g.row_count, 0)::DOUBLE / t.total_rows AS population_share,
           g.observed_default_rate, g.observed_repayment_rate, g.min_p_default, g.max_p_default,
           g.mean_model_p_default,
           g.observed_default_rate - g.mean_model_p_default AS observed_minus_model_default_rate,
           g.observed_default_rate / nullif(t.population_default_rate, 0) AS default_lift,
           g.observed_repayment_rate / nullif(t.population_repayment_rate, 0) AS repayment_lift,
           CASE WHEN t.total_labeled_rows > 0 THEN coalesce(g.default_rows, 0)::DOUBLE
                     / nullif(t.total_defaults, 0) END AS default_capture_share,
           t.total_defaults, t.total_labeled_rows
    FROM totals t CROSS JOIN range(1, 11) d(risk_decile)
    LEFT JOIN grouped g USING (dataset, model_version, risk_decile)
)
SELECT * EXCLUDE (total_defaults, total_labeled_rows),
       CASE WHEN total_labeled_rows > 0 THEN
           sum(default_rows) OVER (PARTITION BY dataset, model_version ORDER BY risk_decile DESC
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::DOUBLE
               / nullif(total_defaults, 0) END AS cumulative_default_capture_from_high_risk
FROM complete;
