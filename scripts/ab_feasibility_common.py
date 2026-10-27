"""Disagreement-only ITT experiment, with full-eligible-population estimands.

All outcomes are historical, fixed-label cash-flow simulations. No model fitting.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import aa_validation_common as aa
    from . import experiment_design_common as legacy
except ImportError:
    import aa_validation_common as aa
    import experiment_design_common as legacy

business = legacy.business
ROOT = business.ROOT
CONFIG = ROOT / "configs/ab_feasibility_v2.json"
SQL = ROOT / "sql/06_ab_feasibility/01_analysis_tables.sql"


def paths(name="ab_feasibility_v2"):
    return business.Paths(root=ROOT, run_name=name)


def load_config(path=CONFIG):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c["scope"] != "offline_feasibility_redesign_and_simulation_only" or c["production_safety_validated"] or not c["champion_retained"]:
        raise ValueError("Offline design retains Champion")
    if c["fraction_b"] != .5 or not c["fixed_final_analysis_only"]:
        raise ValueError("This certified protocol requires 1:1 and a fixed final look")
    if c["business_profit_threshold"] != 5 or c["default_incidence_ni_margin"] != .002 or c["approval_rate_ni_margin"] != .02 or c["absolute_default_rate_limit_approved"] != .05:
        raise ValueError("Do not relax the parent business or risk thresholds")
    if c["planning_profit_alternative"] <= c["business_profit_threshold"] or c["variance_safety_factor"] < 1:
        raise ValueError("Need a positive gap to the business boundary and conservative variance")
    if c["calibration_repetitions"] < 10000 or c["simulation_repetitions"] < 1000 or c["hash_repetitions"] < 2:
        raise ValueError("Insufficient simulation precision")
    if any(not 0 < v <= 1 for v in c["candidate_disagreement_exposures"]):
        raise ValueError("Invalid disagreement exposure")
    if not 0 < c["alpha_two_sided"] < .5 or not .5 < c["target_power"] < 1:
        raise ValueError("Invalid inference contract")
    return c


def cohort(population, name):
    return population.loc[population.policy_cohort.eq(name) & population.eligible_proxy].copy()


def enrich(source):
    legacy.validate_population(source)
    p = source.copy()
    np.testing.assert_array_equal(p.decision_disagreement,p.approved_a.ne(p.approved_b))
    np.testing.assert_array_equal(p.eligible_proxy,p[["score_a","score_b"]].min(axis=1).ge(.90))
    np.testing.assert_array_equal(p.common_score,p[["score_a","score_b"]].min(axis=1))
    codes = np.searchsorted(legacy.load_config()["eligibility"]["risk_stratum_edges"],p.common_score,side="right")-1
    expected_strata = np.where(p.eligible_proxy,[f"S{i+1}" for i in codes],"excluded")
    np.testing.assert_array_equal(p.risk_stratum,expected_strata)
    p["trial_eligible"] = p.eligible_proxy & p.decision_disagreement
    direction = np.where(p.approved_a, "A_approves_B_rejects", "A_rejects_B_approves")
    p["analysis_stratum"] = np.where(p.trial_eligible, p.risk_stratum + "_" + direction, "agreement_or_excluded")
    # Identity premise: identical decisions, terms and cashflows outside D.
    same = p.dataset.eq("train") & ~p.decision_disagreement
    np.testing.assert_array_equal(p.loc[same, "potential_profit_a"], p.loc[same, "potential_profit_b"])
    # Verify the inherited payoff schedule, without inventing borrower fields.
    train = p.dataset.eq("train")
    for arm in ("a", "b"):
        expected = np.where(p.loc[train, f"approved_{arm}"], np.where(p.loc[train, "label"].eq(1), 690., -8510.), -10.)
        np.testing.assert_array_equal(p.loc[train, f"potential_profit_{arm}"], expected)
    return p


def profile(rows):
    """Weights are pre-treatment counts; labels only supply conditional variance."""
    if rows.empty: raise ValueError("Empty target cohort")
    d = rows.loc[rows.decision_disagreement]
    records = []
    for key, h in d.groupby("analysis_stratum", sort=True):
        if h.approved_a.nunique() != 1 or h.approved_b.nunique() != 1: raise ValueError("Nonhomogeneous direction stratum")
        records.append({"stratum":key,"n":len(h),"weight":len(h)/len(rows),
            "approved_a":bool(h.approved_a.iloc[0]),"approved_b":bool(h.approved_b.iloc[0]),
            "bad_labels":int(h.label.eq(0).sum()) if h.label.notna().all() else None,
            "bad_probability":float(h.label.eq(0).mean()) if h.label.notna().all() else np.nan,
            "variance_a":float(h.potential_profit_a.var(ddof=1)),"variance_b":float(h.potential_profit_b.var(ddof=1))})
    frame = pd.DataFrame(records)
    agreement = rows.loc[~rows.decision_disagreement & rows.approved_a]
    meta = {"n_eligible":len(rows),"n_disagreement":len(d),"disagreement_fraction":len(d)/len(rows),
        "agreement_approved_count":len(agreement),"agreement_approved_weight":len(agreement)/len(rows),
        "agreement_bad_labels":int(agreement.label.eq(0).sum()) if agreement.label.notna().all() else None,
        "agreement_bad_probability":float(agreement.label.eq(0).mean()) if len(agreement) and agreement.label.notna().all() else np.nan,
        "approval_delta":float(rows.approved_b.mean()-rows.approved_a.mean()),
        "approval_rate_b":float(rows.approved_b.mean())}
    return frame, meta


def plan(population, config, parent_config):
    dev = cohort(population, "policy_development")
    frame, meta = profile(dev)
    if len(frame) == 0 or frame.n.min() < 10: raise ValueError("Insufficient development-only strata")
    d = meta["disagreement_fraction"]
    conditional = frame.weight.to_numpy()/d
    coefficient = float(np.sum(frame.weight.to_numpy()**2 * (2*frame.variance_a + 2*frame.variance_b) / conditional))
    safe = coefficient * config["variance_safety_factor"]
    gap = config["planning_profit_alternative"] - config["business_profit_threshold"]
    alpha = config["alpha_two_sided"]/2
    primary = legacy.required_sample(safe/4, safe/4, .5, gap, alpha, config["target_power"], True)
    default = legacy.required_sample(safe/(4*9200**2), safe/(4*9200**2), .5,
        config["default_incidence_ni_margin"], alpha, config["target_power"], True)
    cell_minimum = int(np.ceil(2*config["minimum_per_arm_per_stratum"]/conditional.min()))
    n = max(primary, default, cell_minimum)
    old_power, old_n = legacy.planning(population, parent_config)
    rows = [{"design":"original_all_eligible","disagreement_exposure":0.5,
        "required_randomized":old_n,"expected_eligible_arrivals":old_n/.5,
        "enrollment_days":int(np.ceil(old_n/(config["primary_daily_applications"]*config["primary_eligible_fraction"]*.5))),
        "randomized_fraction_of_eligible":.5,"challenger_routed_fraction_of_eligible":.25,
        "changed_decision_fraction_of_eligible":.25*d,
        "business_boundary":5.,"planning_true_gain":10.,"planning_gap":gap,
        "planning_cohort":"policy_development","selected":False}]
    for exposure in config["candidate_disagreement_exposures"]:
        days = int(np.ceil(n/(config["primary_daily_applications"]*config["primary_eligible_fraction"]*d*exposure)))
        rows.append({"design":"disagreement_stratified","disagreement_exposure":exposure,
            "required_randomized":n,"expected_eligible_arrivals":n/(d*exposure),"enrollment_days":days,
            "randomized_fraction_of_eligible":d*exposure,"challenger_routed_fraction_of_eligible":d*exposure/2,
            "changed_decision_fraction_of_eligible":d*exposure/2,
            "business_boundary":5.,"planning_true_gain":10.,"planning_gap":gap,
            "planning_cohort":"policy_development","selected":False})
    candidates = pd.DataFrame(rows)
    candidates["enrollment_feasible"] = candidates.enrollment_days.le(config["max_enrollment_days"])
    options = candidates.loc[candidates.design.eq("disagreement_stratified")]
    feasible = options.loc[options.enrollment_feasible]
    chosen = (feasible if len(feasible) else options).sort_values("disagreement_exposure",ascending=bool(len(feasible))).iloc[0]
    candidates.loc[chosen.name,"selected"] = True
    protocol = {"required_randomized_disagreements":n,"profit_required":primary,"default_ni_required":default,
        "minimum_cell_required":cell_minimum,"selected_disagreement_exposure":float(chosen.disagreement_exposure),
        "planning_coefficient":coefficient,"planning_safe_coefficient":safe,
        "planning_disagreement_fraction":d,"planning_true_gain":config["planning_profit_alternative"],
        "business_boundary":config["business_profit_threshold"],"precision_gap":gap,
        "primary_traffic_enrollment_feasible":bool(chosen.enrollment_feasible),
        "planning_weights":frame[["stratum","weight","n"]].to_dict("records"),
        "minimum_expected_cell_per_arm":float(n*conditional.min()/2),
        "planning_approval_ni_feasible":meta["approval_delta"] > -config["approval_rate_ni_margin"],
        "estimand":"full_common_eligible_cohort_mean_profit_B_minus_A",
        "standardization":"contemporaneous_full_cohort_pre_treatment_stratum_counts; agreement_effect_exactly_zero_only_under_identity_contract",
        "decision":"retain_champion_until_business_lower_bound_guardrails_fixed_sample_maturity_and_quality_all_pass",
        "external_preregistration":False,"reused_development_oof":True,"production_rollout_ready":False}
    return frame, meta, candidates, protocol, old_power


def assign(population, config, mode="AB", salt=None):
    if mode not in ("AB","AA"): raise ValueError("Unknown mode")
    p = population.loc[population.dataset.eq("train") & population.policy_cohort.eq("policy_evaluation") & population.trial_eligible].copy()
    p["assignment_uniform"] = [legacy.stable_uniform(d,i,salt or config["randomization_salt"]) for d,i in zip(p.dataset,p.id)]
    p["arm"] = np.where(p.assignment_uniform < .5,"B","A")
    p["experiment_id"] = config["experiment_id"] + ("_aa" if mode=="AA" else "")
    p["assignment_probability_b"] = .5
    p["simulation_mode"] = mode
    use_b = p.arm.eq("B") & (mode=="AB")
    p["assigned_approved"] = np.where(use_b,p.approved_b,p.approved_a)
    p["simulated_observed_net_profit"] = np.where(use_b,p.potential_profit_b,p.potential_profit_a)
    p["simulated_observed_default"] = p.assigned_approved & p.label.eq(0)
    p["simulated_observed_repaid"] = p.assigned_approved & p.label.eq(1)
    p["observed_dpd30"] = pd.Series(pd.NA,index=p.index,dtype="boolean")
    return p.sort_values(["dataset","id"]).reset_index(drop=True)


def validate_assignments(population, assignments, config, mode="AB", salt=None):
    expected = assign(population,config,mode,salt)
    columns = list(expected.columns)
    pd.testing.assert_frame_equal(assignments[columns].sort_values(["dataset","id"]).reset_index(drop=True), expected,
        check_dtype=False, check_exact=True)
    if assignments.duplicated(["dataset","id"]).any() or assignments.observed_dpd30.notna().any():
        raise AssertionError("Duplicate application or invented DPD")


def infer(n_a,n_b,mean_a,mean_b,var_a,var_b,weights,alpha=.05,boundary=0.,normal=False):
    na,nb = np.asarray(n_a,dtype=float),np.asarray(n_b,dtype=float)
    weights = np.asarray(weights,dtype=float)
    delta = (weights*(np.asarray(mean_b)-np.asarray(mean_a))).sum(axis=-1)
    components_a,components_b = weights**2*np.asarray(var_a)/na, weights**2*np.asarray(var_b)/nb
    variance = (components_a+components_b).sum(axis=-1)
    se = np.sqrt(variance)
    with np.errstate(divide="ignore",invalid="ignore"):
        degrees = variance**2/(components_a**2/(na-1)+components_b**2/(nb-1)).sum(axis=-1)
        z = (delta-boundary)/se
        critical = stats.norm.ppf(1-alpha/2) if normal else stats.t.ppf(1-alpha/2,degrees)
        p = 2*stats.norm.sf(np.abs(z)) if normal else 2*stats.t.sf(np.abs(z),degrees)
    usable = (se>0) & np.isfinite(p) & (na>=2).all(axis=-1) & (nb>=2).all(axis=-1)
    return {"delta":delta,"se":se,"df":degrees,"ci_low":delta-critical*se,"ci_high":delta+critical*se,"p":p,"usable":usable}


def estimate(assignments, target_rows, config, mode="AB"):
    frame, meta = profile(target_rows)
    # No approval/default/label-based exclusions after randomization.
    records = []
    for row in frame.itertuples(index=False):
        for arm in ("A","B"):
            group = assignments.loc[assignments.analysis_stratum.eq(row.stratum) & assignments.arm.eq(arm)]
            if len(group)<2: raise ValueError("Insufficient assigned cell; no significance claim")
            records.append({"stratum":row.stratum,"arm":arm,"n":len(group),"weight":row.weight,
                "profit_mean":float(group.simulated_observed_net_profit.mean()),"profit_variance":float(group.simulated_observed_net_profit.var(ddof=1)),
                "bad_mean":float(group.simulated_observed_default.mean()),"bad_variance":float(group.simulated_observed_default.var(ddof=1)),
                "approved":int(group.assigned_approved.sum()),"bad":int(group.simulated_observed_default.sum())})
    cells = pd.DataFrame(records)
    a,b = [cells.loc[cells.arm.eq(arm)].sort_values("stratum") for arm in ("A","B")]
    weights = a.weight.to_numpy()
    profit = infer(a.n.to_numpy(),b.n.to_numpy(),a.profit_mean.to_numpy(),b.profit_mean.to_numpy(),
        a.profit_variance.to_numpy(),b.profit_variance.to_numpy(),weights,config["alpha_two_sided"])
    bad = infer(a.n.to_numpy(),b.n.to_numpy(),a.bad_mean.to_numpy(),b.bad_mean.to_numpy(),
        a.bad_variance.to_numpy(),b.bad_variance.to_numpy(),weights,config["alpha_two_sided"],normal=True)
    result = {name:float(profit[name]) for name in ("delta","se","df","ci_low","ci_high","p")}
    result["default_delta"] = float(bad["delta"])
    result["default_upper"] = float(bad["ci_high"])
    result["approval_delta"] = meta["approval_delta"] if mode=="AB" else 0.
    result["business_threshold"] = config["business_profit_threshold"]
    result["business_value_demonstrated_in_simulation"] = bool(profit["ci_low"]>config["business_profit_threshold"])
    result["positive_zero_ci_in_simulation"] = bool(profit["ci_low"]>0)
    result["default_ni_passed"] = bool(bad["ci_high"]<config["default_incidence_ni_margin"])
    result["approval_ni_passed"] = bool(result["approval_delta"]>-config["approval_rate_ni_margin"])
    active = frame.approved_b.to_numpy() if mode=="AB" else frame.approved_a.to_numpy()
    upper = absolute_risk_upper(b.bad.to_numpy(),b.n.to_numpy(),active,weights,
        meta["agreement_bad_labels"],meta["agreement_approved_count"],meta["agreement_approved_weight"],
        meta["approval_rate_b"] if mode=="AB" else float(target_rows.approved_a.mean()),config["alpha_two_sided"]/2)
    result["full_portfolio_approved_default_upper"] = float(upper)
    result["absolute_risk_passed"] = bool(upper<=config["absolute_default_rate_limit_approved"])
    result["minimum_cell_sample_reached"] = bool(cells.n.ge(config["minimum_per_arm_per_stratum"]).all())
    # Cash-flow identity is an independent cross-endpoint audit.
    np.testing.assert_allclose(result["delta"],700*result["approval_delta"]-9200*result["default_delta"],atol=1e-8,rtol=0)
    return result,cells


def binomial_upper(k,n,error):
    k,n = np.asarray(k),np.asarray(n)
    with np.errstate(invalid="ignore"):
        return np.where(k>=n,1.,stats.beta.ppf(1-error,k+1,n-k))


def absolute_risk_upper(k_b,n_b,active,weights,k_ag,n_ag,w_ag,approval_b,error):
    if approval_b<=0 or n_ag<=0: return np.nan
    active = np.asarray(active,dtype=bool)
    local = error/(int(active.sum())+1)
    bounds = binomial_upper(np.asarray(k_b)[...,active],np.asarray(n_b)[...,active],local)
    upper_events = (bounds*np.asarray(weights)[active]).sum(axis=-1)+w_ag*binomial_upper(k_ag,n_ag,local)
    return upper_events/approval_b


def quality(assignments,config):
    rows = []
    groups = [("all",assignments),*list(assignments.groupby("analysis_stratum",sort=True))]
    for key,g in groups:
        nb = int(g.arm.eq("B").sum())
        rows.append({"stratum":key,"n":len(g),"n_a":len(g)-nb,"n_b":nb,
            "srm_p":float(stats.binomtest(nb,len(g),.5).pvalue),"srm_familywise_cutoff":.001/len(groups)})
    frame = pd.DataFrame(rows)
    frame["passed"] = frame.srm_p.ge(frame.srm_familywise_cutoff)
    return frame


def rng(config,domain):
    return aa.rng_for(config,"feasibility_v2:"+domain)


def counts_from_total(frame,total):
    conditional = frame.n.to_numpy()/frame.n.sum()
    raw = total*conditional
    counts = np.floor(raw).astype(int)
    missing = total-int(counts.sum())
    order = np.argsort(-(raw-counts),kind="stable")
    counts[order[:missing]] += 1
    return counts


def simulate(frame,meta,total,repetitions,config,domain,mode="AB",finite=False,target_profit=None,target_default=None,exposure=1.):
    """Cell counts exactly reproduce IID binary historical cash-flow marginals.

    Synthetic alternatives change funded B default probabilities, never actual
    replay labels or returned policy predictions. They are not A/A evidence.
    """
    generator = rng(config,domain)
    weights = frame.weight.to_numpy()
    pa,pb = frame.approved_a.to_numpy(dtype=float),frame.approved_b.to_numpy(dtype=float)
    q_a,q_b = frame.bad_probability.to_numpy(),frame.bad_probability.to_numpy().copy()
    if mode=="AA": pb = pa.copy()
    approval_delta = float(weights@(pb-pa))
    true_default = float(weights@(pb*q_b-pa*q_a))
    actual_profit = 700*approval_delta-9200*true_default
    if target_profit is not None or target_default is not None:
        if mode!="AB" or finite: raise ValueError("Synthetic boundary is not genuine A/A")
        target = target_default if target_default is not None else (700*approval_delta-target_profit)/9200
        adjustable = pb.astype(bool)
        q_b[adjustable] += (target-true_default)/float(weights@pb)
        if ((q_b<0)|(q_b>1)).any(): raise ValueError("Infeasible binary-outcome alternative")
    counts = frame.n.to_numpy(dtype=int) if finite else counts_from_total(frame,total)
    if min(counts)<4: raise ValueError("Too few rows in a stratum")
    if finite:
        bad_b = generator.binomial(frame.bad_labels.to_numpy(dtype=int),.5,size=(repetitions,len(frame)))
        good_b = generator.binomial(counts-frame.bad_labels.to_numpy(dtype=int),.5,size=(repetitions,len(frame)))
        nb,na = bad_b+good_b,counts-(bad_b+good_b)
        bad_a = frame.bad_labels.to_numpy(dtype=int)-bad_b
    else:
        nb = generator.binomial(counts,.5,size=(repetitions,len(frame)))
        na = counts-nb
        bad_a,bad_b = generator.binomial(na,q_a),generator.binomial(nb,q_b)
    with np.errstate(divide="ignore",invalid="ignore"):
        qa,qb = bad_a/na,bad_b/nb
        ma,mb = -10+pa*(700-9200*qa),-10+pb*(700-9200*qb)
        va,vb = pa*9200**2*qa*(1-qa)*na/(na-1),pb*9200**2*qb*(1-qb)*nb/(nb-1)
        profit = infer(na,nb,ma,mb,va,vb,weights,config["alpha_two_sided"])
        ba,bb = pa*qa,pb*qb
        bad = infer(na,nb,ba,bb,pa*qa*(1-qa)*na/(na-1),pb*qb*(1-qb)*nb/(nb-1),weights,config["alpha_two_sided"],normal=True)
    true_bad = float(weights@(pb*q_b-pa*q_a))
    true_gain = 700*approval_delta-9200*true_bad
    n_ag = max(1,int(np.floor(total/(meta["disagreement_fraction"]*exposure)*meta["agreement_approved_weight"])))
    k_ag = generator.binomial(n_ag,meta["agreement_bad_probability"],size=repetitions)
    approval_b = meta["agreement_approved_weight"]+float(weights@pb)
    upper = absolute_risk_upper(bad_b*pb,nb,pb.astype(bool),weights,k_ag,n_ag,
        meta["agreement_approved_weight"],approval_b,config["alpha_two_sided"]/2)
    usable = profit["usable"] & bad["usable"]
    return pd.DataFrame({"replicate":np.arange(repetitions),"profit_delta":profit["delta"],"profit_se":profit["se"],
        "profit_df":profit["df"],"profit_ci_low":profit["ci_low"],"profit_ci_high":profit["ci_high"],
        "profit_p_zero":profit["p"],"default_delta":bad["delta"],"default_se":bad["se"],
        "default_upper":bad["ci_high"],"default_p_zero":bad["p"],"absolute_default_upper":upper,
        "business_success":profit["ci_low"]>config["business_profit_threshold"],
        "positive_zero_success":profit["ci_low"]>0,"default_ni_success":bad["ci_high"]<config["default_incidence_ni_margin"],
        "approval_ni_success":approval_delta>-config["approval_rate_ni_margin"],
        "absolute_risk_success":upper<=config["absolute_default_rate_limit_approved"],
        "usable":usable,"true_profit":true_gain,"true_default":true_bad,"total_disagreements":int(counts.sum())})


def calibrate(population,protocol,config,progress=None):
    evaluation, em = profile(cohort(population,"policy_evaluation"))
    development, dm = profile(cohort(population,"policy_development"))
    n = protocol["required_randomized_disagreements"]
    definitions = [("aa_current_finite",evaluation,em,int(evaluation.n.sum()),"AA",True,None,None),
        ("aa_current_iid",evaluation,em,int(evaluation.n.sum()),"AA",False,None,None),
        ("aa_planned_iid",development,dm,n,"AA",False,None,None),
        ("business_boundary_5",development,dm,n,"AB",False,5.,None),
        ("default_ni_boundary",development,dm,n,"AB",False,None,.002)]
    comparisons = 3*2*len(config["calibration_alpha_grid"])+2
    error = config["calibration_familywise_error"]/comparisons
    records, draws = [],[]
    for case,frame,meta,total,mode,finite,gain,bad_target in definitions:
        sim = simulate(frame,meta,total,config["calibration_repetitions"],config,"calibration:"+case,mode,finite,gain,bad_target)
        sim.insert(0,"case_id",case)
        draws.append(sim)
        tests = [(endpoint,alpha,sim[field].lt(alpha)) for endpoint,field in (("profit_equality","profit_p_zero"),("default_equality","default_p_zero"))
            for alpha in config["calibration_alpha_grid"]] if mode=="AA" else [("business_boundary" if gain is not None else "default_ni_boundary",.025,
                sim.business_success if gain is not None else sim.default_ni_success)]
        for endpoint,alpha,rejected in tests:
            usable = sim.usable
            count,total_valid = int((rejected & usable).sum()),int(usable.sum())
            low,high = aa.clopper_pearson(count,total_valid,error)
            tolerance = config["calibration_tolerances"][str(alpha)]
            records.append({"case_id":case,"endpoint":endpoint,"alpha":alpha,"repetitions":len(sim),
                "valid_repetitions":total_valid,"rejections":count,"false_positive_rate":count/total_valid,
                "simultaneous_ci_low":low,"simultaneous_ci_high":high,"acceptance_low":alpha-tolerance,
                "acceptance_high":alpha+tolerance,"passed":bool(total_valid==len(sim) and low>=alpha-tolerance and high<=alpha+tolerance),
                "genuine_same_champion_aa":mode=="AA","synthetic_boundary":mode!="AA"})
        if progress: progress(f"Matched calibration {case}: {len(sim):,} repetitions; certified={all(r['passed'] for r in records if r['case_id']==case)}")
    return pd.DataFrame(records),pd.concat(draws,ignore_index=True)


def hash_replays(population,config,progress=None):
    target = cohort(population,"policy_evaluation")
    rows = []
    for i in range(config["hash_repetitions"]):
        salt = config["randomization_salt"]+f":aa:{i}"
        assigned = assign(population,config,"AA",salt)
        validate_assignments(population,assigned,config,"AA",salt)
        est,_ = estimate(assigned,target,config,"AA")
        q = quality(assigned,config)
        rows.append({"replicate":i,"salt":salt,"delta":est["delta"],"se":est["se"],"ci_low":est["ci_low"],
            "ci_high":est["ci_high"],"p":est["p"],"srm_family_passed":bool(q.passed.all())})
        if progress and (i+1)%8==0: progress(f"Matched real A/A hash replays {i+1}/{config['hash_repetitions']}")
    return pd.DataFrame(rows)


def simulations(population,protocol,config,progress=None):
    frame,meta = profile(cohort(population,"policy_evaluation"))
    dev,dm = profile(cohort(population,"policy_development"))
    n = protocol["required_randomized_disagreements"]
    definitions = [("historical_current",frame,meta,int(frame.n.sum()),None),
        ("historical_at_planned",frame,meta,n,None)]
    definitions += [(f"synthetic_true_gain_{gain:g}",dev,dm,n,gain) for gain in config["simulation_profit_alternatives"]]
    records,draws = [],[]
    for case,f,m,total,gain in definitions:
        sim = simulate(f,m,total,config["simulation_repetitions"],config,"ab:"+case,target_profit=gain,
            exposure=protocol["selected_disagreement_exposure"])
        if not sim.usable.all(): raise AssertionError("Invalid simulation cell; cannot certify feasibility")
        joint = sim.business_success & sim.default_ni_success & sim.approval_ni_success & sim.absolute_risk_success
        row = {"case_id":case,"n_disagreements":total,"true_profit":float(sim.true_profit.iloc[0]),
            "repetitions":len(sim),"synthetic_alternative":gain is not None,"mean_estimate":float(sim.profit_delta.mean()),
            "median_ci_width":float((sim.profit_ci_high-sim.profit_ci_low).median())}
        for endpoint,events in (("business_value",sim.business_success),("positive_zero",sim.positive_zero_success),
            ("default_ni",sim.default_ni_success),("approval_ni",sim.approval_ni_success),
            ("absolute_risk",sim.absolute_risk_success),("joint_success",joint)):
            low,high = aa.clopper_pearson(int(events.sum()),len(sim))
            row[endpoint+"_rate"] = float(events.mean())
            row[endpoint+"_mc_low"],row[endpoint+"_mc_high"] = low,high
        records.append(row)
        sim.insert(0,"case_id",case)
        draws.append(sim)
        if progress: progress(f"A/B {case}: business success={row['business_value_rate']:.2%}; joint={row['joint_success_rate']:.2%}")
    return pd.DataFrame(records),pd.concat(draws,ignore_index=True)


def bootstrap(assignments,target,config):
    estimate_result,cells = estimate(assignments,target,config)
    generator = rng(config,"stratified_independent_bootstrap")
    deltas = np.zeros(config["bootstrap_repetitions"])
    for cell in cells.itertuples(index=False):
        support = assignments.loc[assignments.analysis_stratum.eq(cell.stratum) & assignments.arm.eq(cell.arm),"simulated_observed_net_profit"]
        deltas += (1 if cell.arm=="B" else -1)*cell.weight*legacy.bootstrap_mean(support,len(deltas),generator)
    low,high = np.quantile(deltas,[.025,.975])
    return pd.DataFrame([{"method":"independent_within_stratum_full_population_weighted",
        "estimate":estimate_result["delta"],"ci_low":low,"ci_high":high,"repetitions":len(deltas)}]),pd.DataFrame({"replicate":np.arange(len(deltas)),"profit_delta":deltas})


def duration(population,protocol,config):
    records = []
    fractions = [("planning_development",protocol["planning_disagreement_fraction"])]
    test,tm = profile(cohort(population,"unlabeled_test"))
    fractions.append(("unlabeled_test_covariate_sensitivity",tm["disagreement_fraction"]))
    for label,d in fractions:
        for daily in config["daily_applications_grid"]:
            for eligibility in config["eligible_fraction_grid"]:
                rate = daily*eligibility*d*protocol["selected_disagreement_exposure"]
                days = int(np.ceil(protocol["required_randomized_disagreements"]/rate))
                records.append({"coverage_source":label,"daily_applications":daily,"eligible_fraction":eligibility,
                    "disagreement_fraction":d,"disagreement_exposure":protocol["selected_disagreement_exposure"],
                    "expected_daily_randomized":rate,"enrollment_days":days,
                    "maturity_days":config["outcome_maturity_days_assumed"],"grace_days":config["data_grace_days"],
                    "final_analysis_days":days+config["outcome_maturity_days_assumed"]+config["data_grace_days"],
                    "feasible_enrollment":days<=config["max_enrollment_days"]})
    return pd.DataFrame(records)


def input_hashes(config):
    files = [CONFIG,ROOT/"configs/ab_test_design.json",
        ROOT/f"artifacts/{config['parent_run_name']}/experiment_freeze_manifest.json",
        ROOT/f"artifacts/{config['parent_run_name']}/experiment_population.parquet",
        ROOT/f"artifacts/{config['parent_run_name']}/experiment_protocol.json",
        ROOT/f"artifacts/{config['aa_parent_run_name']}/aa_freeze_manifest.json",
        ROOT/f"reports/{config['aa_parent_run_name']}/artifact_verification.json"]
    return {business.relative(p,ROOT):business.sha256(p) for p in files}


def protocol_hashes():
    files = [Path(__file__),ROOT/"scripts/run_ab_feasibility.py",ROOT/"scripts/verify_ab_feasibility.py",
        ROOT/"scripts/experiment_design_common.py",ROOT/"scripts/aa_validation_common.py",
        ROOT/"scripts/verify_ab_test_design.py",ROOT/"scripts/verify_aa_validation.py",SQL]
    return {business.relative(p,ROOT):business.sha256(p) for p in files}


def check_manifest(output,config):
    path = output.artifacts/"feasibility_freeze_manifest.json"
    if not path.exists(): return None
    m = json.loads(path.read_text(encoding="utf-8"))
    if m["input_sha256"]!=input_hashes(config) or m["protocol_sha256"]!=protocol_hashes():
        raise AssertionError("Frozen input/protocol changed; use a new run-name")
    for name,digest in m["output_sha256"].items():
        if business.sha256(ROOT/name)!=digest: raise AssertionError("Frozen output changed")
    return m
