"""Read-only design diagnostics; evaluation outcomes are deliberately not read."""
import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with duckdb.connect() as con:
    con.read_parquet(str(ROOT / "artifacts/ab_test_design/experiment_population.parquet")).create_view("p")
    print(con.execute("""SELECT dataset, policy_cohort, count(*) AS all_rows,
      count(*) FILTER(WHERE eligible_proxy) AS eligible,
      count(*) FILTER(WHERE eligible_proxy AND decision_disagreement) AS disagreement
      FROM p GROUP BY ALL ORDER BY dataset,policy_cohort""").fetchdf().to_string(index=False))
    print(con.execute("""SELECT risk_stratum,approved_a,approved_b,count(*) AS n,
      avg(potential_profit_a) AS mean_a,avg(potential_profit_b) AS mean_b,
      var_samp(potential_profit_a) AS var_a,var_samp(potential_profit_b) AS var_b,
      avg((approved_a AND label=0)::INTEGER) AS bad_a,
      avg((approved_b AND label=0)::INTEGER) AS bad_b
      FROM p WHERE policy_cohort='policy_development' AND eligible_proxy
      GROUP BY ALL ORDER BY risk_stratum,approved_a,approved_b""").fetchdf().to_string(index=False))
