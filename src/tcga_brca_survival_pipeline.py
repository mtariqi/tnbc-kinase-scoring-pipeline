"""
TCGA-BRCA Survival Pipeline
===========================

Produces the per-kinase `cox_hr` and `logrank_p` values that compute_cts()
in kinase_scoring_pipeline.py needs -- this is the "Survival" input that
you compute yourself (Section 10 of the proposal), not one fetched from
an external API.

DEPENDENCY: pip install lifelines
    (the standard, validated Python survival-analysis library -- this
    pipeline does not reimplement Cox regression or the Schoenfeld
    residual test from scratch, since a hand-rolled version would be a
    real risk to trust for anything you'd put in a proposal or paper.
    Log-rank testing has one manual fallback path, see below.)

REAL DATA THIS PIPELINE EXPECTS
--------------------------------
1. Clinical data (from GDC TCGA-BRCA clinical supplement or cBioPortal),
   one row per patient, with AT LEAST:
       patient_id, vital_status ('Alive'/'Dead'),
       days_to_death, days_to_last_follow_up,
       ajcc_pathologic_stage (or tumor_stage), age_at_diagnosis,
       purity (e.g. from ABSOLUTE/ESTIMATE), tmb (mutations/Mb)

2. Expression matrix (from GDC RNA-seq, TPM or log2(TPM+1)):
       rows = patient_id, columns = gene symbols (your 90 RTK/NRTK kinases
       at minimum; more is fine, this pipeline only touches the columns
       you ask it to score)

3. (Optional, for external validation per Section 10) The same two
   tables for GSE58812 / GSE76275 / GSE135565 pulled from GEO, re-
   normalized to TPM and ComBat-corrected against the TCGA-BRCA batch
   before combining -- see `run_external_validation()` at the bottom for
   where that step plugs in (ComBat itself is not implemented here; use
   the `pycombat` or `combat-seq` packages for that step specifically,
   it is a distinct, well-solved problem from survival analysis).

OUTPUT
------
`run_survival_pipeline(...)` returns a DataFrame indexed by kinase_id with
columns: cox_hr, cox_hr_ci_low, cox_hr_ci_high, cox_p, logrank_p,
rmst_diff_months, ph_assumption_ok -- feed the cox_hr/logrank_p columns
directly into kinase_scoring_pipeline.compute_cts().
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, proportional_hazard_test
    from lifelines.utils import restricted_mean_survival_time
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


# =====================================================================
# 1. DERIVE SURVIVAL TIME + EVENT FROM RAW TCGA CLINICAL FIELDS
# =====================================================================

def derive_survival_time_event(
    clinical_df: pd.DataFrame,
    time_unit: str = "months",
) -> pd.DataFrame:
    """
    Standard TCGA convention: for deceased patients use days_to_death; for
    living patients use days_to_last_follow_up (right-censored).

    Returns clinical_df with two added columns: time (in `time_unit`), event
    (1 = death observed, 0 = censored).
    """
    df = clinical_df.copy()
    is_dead = df["vital_status"].str.lower().eq("dead")

    days = np.where(is_dead, df["days_to_death"], df["days_to_last_follow_up"])
    df["event"] = is_dead.astype(int)

    divisor = {"days": 1, "months": 30.4375, "years": 365.25}[time_unit]
    df["time"] = pd.to_numeric(pd.Series(days, index=df.index), errors="coerce") / divisor

    n_missing = df["time"].isna().sum()
    if n_missing:
        warnings.warn(f"{n_missing} patients have no usable time-to-event value and will be dropped.")
    return df.dropna(subset=["time"])


# =====================================================================
# 2. MEDIAN-EXPRESSION STRATIFICATION (per Section 10)
# =====================================================================

def median_stratify(expression: pd.Series) -> pd.Series:
    """Return a 'high'/'low' Series based on a median split of one gene's
    expression across patients. Ties at the median are assigned 'low'."""
    median_val = expression.median()
    return pd.Series(np.where(expression > median_val, "high", "low"), index=expression.index)


# =====================================================================
# 3. LOG-RANK TEST (Kaplan-Meier, high vs low expression)
# =====================================================================

def logrank_by_expression(
    time: pd.Series,
    event: pd.Series,
    expression: pd.Series,
) -> float:
    """
    Log-rank p-value comparing KM curves for high- vs low-expression groups
    (median split). Uses lifelines if available (recommended -- handles
    ties correctly); falls back to a manual Mantel-Haenszel log-rank
    implementation otherwise (equivalent for the no-ties/few-ties case,
    but lifelines is preferred for real analysis).
    """
    group = median_stratify(expression)
    high_mask = group == "high"

    if LIFELINES_AVAILABLE:
        result = logrank_test(
            time[high_mask], time[~high_mask],
            event_observed_A=event[high_mask], event_observed_B=event[~high_mask],
        )
        return float(result.p_value)

    return _manual_logrank_p(
        time[high_mask].values, event[high_mask].values,
        time[~high_mask].values, event[~high_mask].values,
    )


def _manual_logrank_p(t_a, e_a, t_b, e_b) -> float:
    """Standard two-sample log-rank test (Mantel-Haenszel), no external
    dependency. Used only when lifelines is not installed."""
    from scipy.stats import chi2

    all_times = np.sort(np.unique(np.concatenate([t_a[e_a == 1], t_b[e_b == 1]])))
    O_A = E_A = V_A = 0.0
    for t in all_times:
        n_a = np.sum(t_a >= t)
        n_b = np.sum(t_b >= t)
        n = n_a + n_b
        if n <= 1:
            continue
        d_a = np.sum((t_a == t) & (e_a == 1))
        d_b = np.sum((t_b == t) & (e_b == 1))
        d = d_a + d_b
        if d == 0:
            continue
        e_a_t = d * n_a / n
        var_t = (d * (n_a / n) * (n_b / n) * (n - d)) / (n - 1) if n > 1 else 0.0
        O_A += d_a
        E_A += e_a_t
        V_A += var_t
    if V_A == 0:
        return 1.0
    chi_sq_stat = (O_A - E_A) ** 2 / V_A
    return float(1 - chi2.cdf(chi_sq_stat, df=1))


# =====================================================================
# 4. MULTIVARIABLE COX PH (adjusting for stage, age, purity, TMB)
# =====================================================================

def cox_hr_for_gene(
    merged_df: pd.DataFrame,
    gene: str,
    adjust_for: Iterable[str] = ("stage_numeric", "age", "purity", "tmb"),
) -> dict:
    """
    Fit a multivariable Cox model: gene expression + covariates -> hazard.
    `merged_df` must already have 'time', 'event', the gene's expression
    column, and every column named in `adjust_for`.

    Returns dict: hr, hr_ci_low, hr_ci_high, p, concordance.
    Requires lifelines (raises ImportError with an install hint otherwise).
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError(
            "Cox regression requires lifelines. Run: pip install lifelines"
        )

    cols = ["time", "event", gene] + list(adjust_for)
    model_df = merged_df[cols].dropna()

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="time", event_col="event")

    summary = cph.summary.loc[gene]
    return {
        "hr": float(np.exp(summary["coef"])),
        "hr_ci_low": float(np.exp(summary["coef lower 95%"])),
        "hr_ci_high": float(np.exp(summary["coef upper 95%"])),
        "p": float(summary["p"]),
        "concordance": float(cph.concordance_index_),
        "_fitted_model": cph,  # kept for the Schoenfeld test below
    }


def check_proportional_hazards(cph_result: dict, merged_df: pd.DataFrame) -> bool:
    """
    Grambsch-Therneau (scaled Schoenfeld residual) test via lifelines.
    Returns True if the proportional-hazards assumption holds (p > 0.05
    for all covariates -- i.e. no significant time-dependence detected).
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError("Proportional-hazards testing requires lifelines. Run: pip install lifelines")

    cph = cph_result["_fitted_model"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = proportional_hazard_test(cph, merged_df, time_transform="rank")
    return bool((results.summary["p"] > 0.05).all())


# =====================================================================
# 5. RESTRICTED MEAN SURVIVAL TIME (RMST) DIFFERENCE
# =====================================================================

def rmst_difference(
    time: pd.Series,
    event: pd.Series,
    expression: pd.Series,
    timeline_horizon: Optional[float] = None,
) -> float:
    """RMST(high-expression group) - RMST(low-expression group), in the
    same time unit as `time` (e.g. months). Requires lifelines."""
    if not LIFELINES_AVAILABLE:
        raise ImportError("RMST requires lifelines. Run: pip install lifelines")

    group = median_stratify(expression)
    horizon = timeline_horizon or time.max()

    kmf_high = KaplanMeierFitter().fit(time[group == "high"], event[group == "high"])
    kmf_low = KaplanMeierFitter().fit(time[group == "low"], event[group == "low"])

    rmst_high = restricted_mean_survival_time(kmf_high, t=horizon)
    rmst_low = restricted_mean_survival_time(kmf_low, t=horizon)
    return float(rmst_high - rmst_low)


# =====================================================================
# 6. BATCH RUNNER -- ALL KINASES AT ONCE
# =====================================================================

def run_survival_pipeline(
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    genes: List[str],
    adjust_for: Iterable[str] = ("stage_numeric", "age", "purity", "tmb"),
    time_unit: str = "months",
) -> pd.DataFrame:
    """
    Full per-gene survival analysis, matching Section 10's design:
    median-expression KM + log-rank, multivariable Cox PH, Schoenfeld PH
    check, and RMST difference -- for every gene in `genes`.

    clinical_df : indexed or joinable on patient_id, raw TCGA clinical fields
                  (see module docstring for required columns)
    expression_df : patient_id (rows) x gene symbol (columns), TPM/log2TPM
    genes : which columns of expression_df to score (your 90 kinases)

    Returns a DataFrame indexed by gene with columns:
        cox_hr, cox_hr_ci_low, cox_hr_ci_high, cox_p, logrank_p,
        rmst_diff_months, ph_assumption_ok, cox_concordance
    -- feed cox_hr and logrank_p straight into compute_cts().
    """
    surv_df = derive_survival_time_event(clinical_df, time_unit=time_unit)
    merged = surv_df.join(expression_df, how="inner")

    rows = []
    for gene in genes:
        if gene not in merged.columns:
            warnings.warn(f"Gene '{gene}' not found in expression_df -- skipping.")
            continue

        gene_df = merged.dropna(subset=[gene])
        if gene_df["event"].sum() < 5:
            warnings.warn(f"Gene '{gene}': fewer than 5 events, results unstable -- skipping.")
            continue

        logrank_p = logrank_by_expression(gene_df["time"], gene_df["event"], gene_df[gene])

        row = {"kinase_id": gene, "logrank_p": logrank_p}

        if LIFELINES_AVAILABLE:
            try:
                cox_result = cox_hr_for_gene(gene_df, gene, adjust_for=adjust_for)
                ph_ok = check_proportional_hazards(cox_result, gene_df[["time", "event", gene, *adjust_for]].dropna())
                rmst = rmst_difference(gene_df["time"], gene_df["event"], gene_df[gene])
                row.update(
                    {
                        "cox_hr": cox_result["hr"],
                        "cox_hr_ci_low": cox_result["hr_ci_low"],
                        "cox_hr_ci_high": cox_result["hr_ci_high"],
                        "cox_p": cox_result["p"],
                        "cox_concordance": cox_result["concordance"],
                        "ph_assumption_ok": ph_ok,
                        "rmst_diff_months": rmst,
                    }
                )
            except Exception as exc:  # keep going even if one gene's Cox model fails to converge
                warnings.warn(f"Gene '{gene}': Cox model failed ({exc}); logrank_p still recorded.")
        rows.append(row)

    result = pd.DataFrame(rows).set_index("kinase_id")
    return result


# =====================================================================
# 7. EXTERNAL VALIDATION (GSE58812 / GSE76275 / GSE135565) -- STUB
# =====================================================================

def run_external_validation(
    tcga_result: pd.DataFrame,
    geo_clinical_df: pd.DataFrame,
    geo_expression_df: pd.DataFrame,
    genes: List[str],
    cohort_name: str,
) -> pd.DataFrame:
    """
    Re-run the same survival pipeline on a GEO validation cohort and report
    directional concordance with the TCGA-BRCA discovery result per gene
    (does the sign of the HR agree?).

    NOTE: before calling this, re-normalize the GEO expression matrix to
    TPM and apply ComBat batch correction against the TCGA-BRCA batch
    (e.g. via the `pycombat` package's `pycombat_norm`). That step is
    intentionally NOT implemented here -- batch correction is a distinct,
    well-solved problem from survival analysis and should use a dedicated,
    validated tool rather than a hand-rolled version.
    """
    geo_result = run_survival_pipeline(geo_clinical_df, geo_expression_df, genes)
    geo_result = geo_result.add_prefix(f"{cohort_name}_")

    combined = tcga_result.join(geo_result, how="inner")
    if "cox_hr" in tcga_result.columns and f"{cohort_name}_cox_hr" in geo_result.columns:
        combined["hr_direction_concordant"] = (
            (combined["cox_hr"] > 1) == (combined[f"{cohort_name}_cox_hr"] > 1)
        )
    return combined


# =====================================================================
# 8. SMOKE TESTS -- SYNTHETIC DATA, CLEARLY NOT REAL
# =====================================================================

def _make_synthetic_data(n_patients: int = 200, seed: int = 42, planted_effect: bool = False):
    rng = np.random.default_rng(seed)
    patient_ids = [f"TCGA-{i:04d}" for i in range(n_patients)]

    vital_status = rng.choice(["Alive", "Dead"], n_patients, p=[0.4, 0.6] if planted_effect else [0.7, 0.3])
    is_dead = vital_status == "Dead"

    genes = ["EGFR", "ERBB2", "SRC", "JAK1"]
    expr = rng.normal(5, 2, size=(n_patients, len(genes)))

    if planted_effect:
        # Make high EGFR expression genuinely associated with shorter survival,
        # so we can confirm the log-rank test actually detects a real signal
        # rather than just running without error.
        egfr_col = genes.index("EGFR")
        base_days = rng.integers(200, 3000, n_patients).astype(float)
        high_expr_penalty = np.clip((expr[:, egfr_col] - 5) * 600, 0, None)
        base_days = np.clip(base_days - high_expr_penalty, 30, None)
    else:
        base_days = rng.integers(30, 3000, n_patients).astype(float)

    days_to_death = np.where(is_dead, base_days, np.nan)
    days_to_last_follow_up = np.where(~is_dead, base_days, np.nan)

    clinical_df = pd.DataFrame(
        {
            "vital_status": vital_status,
            "days_to_death": days_to_death,
            "days_to_last_follow_up": days_to_last_follow_up,
            "stage_numeric": rng.integers(1, 5, n_patients),
            "age": rng.integers(30, 85, n_patients),
            "purity": rng.uniform(0.4, 0.95, n_patients),
            "tmb": rng.exponential(2.0, n_patients),
        },
        index=patient_ids,
    )
    clinical_df.index.name = "patient_id"

    expression_df = pd.DataFrame(expr, index=patient_ids, columns=genes)
    expression_df.index.name = "patient_id"

    return clinical_df, expression_df, genes


def _run_smoke_tests() -> None:
    print("=== Testing derive_survival_time_event + median_stratify (no lifelines needed) ===")
    clinical_df, expression_df, genes = _make_synthetic_data()
    surv_df = derive_survival_time_event(clinical_df)
    print(surv_df[["time", "event"]].head(), "\n")
    assert surv_df["event"].isin([0, 1]).all()
    assert (surv_df["time"] > 0).all()

    group = median_stratify(expression_df["EGFR"])
    print(f"EGFR high/low split: {group.value_counts().to_dict()}\n")
    assert set(group.unique()) <= {"high", "low"}

    print("=== Testing manual log-rank fallback: null case (no real signal) ===")
    merged = surv_df.join(expression_df, how="inner")
    p_null = _manual_logrank_p(
        merged.loc[median_stratify(merged["EGFR"]) == "high", "time"].values,
        merged.loc[median_stratify(merged["EGFR"]) == "high", "event"].values,
        merged.loc[median_stratify(merged["EGFR"]) == "low", "time"].values,
        merged.loc[median_stratify(merged["EGFR"]) == "low", "event"].values,
    )
    print(f"  p-value (no planted effect): {p_null:.4f} -- should be unremarkable, not near 0\n")
    assert 0.0 <= p_null <= 1.0

    print("=== Correctness check: log-rank test on data with a DELIBERATELY PLANTED effect ===")
    clinical_planted, expression_planted, _ = _make_synthetic_data(n_patients=300, planted_effect=True)
    surv_planted = derive_survival_time_event(clinical_planted)
    merged_planted = surv_planted.join(expression_planted, how="inner")
    group_planted = median_stratify(merged_planted["EGFR"])
    p_planted = _manual_logrank_p(
        merged_planted.loc[group_planted == "high", "time"].values,
        merged_planted.loc[group_planted == "high", "event"].values,
        merged_planted.loc[group_planted == "low", "time"].values,
        merged_planted.loc[group_planted == "low", "event"].values,
    )
    median_high = merged_planted.loc[group_planted == "high", "time"].median()
    median_low = merged_planted.loc[group_planted == "low", "time"].median()
    print(f"  median survival time -- high-EGFR group: {median_high:.1f}, low-EGFR group: {median_low:.1f}")
    print(f"  p-value (planted effect): {p_planted:.6f}")
    assert median_high < median_low, "planted effect should make high-EGFR die sooner"
    assert p_planted < 0.01, "log-rank test should detect the deliberately planted difference"
    print("  PASSED: high-EGFR group has shorter survival AND the test detects it as significant.\n")

    if LIFELINES_AVAILABLE:
        print("=== lifelines IS installed -- running the full pipeline end-to-end ===")
        result = run_survival_pipeline(clinical_df, expression_df, genes)
        print(result, "\n")
        assert "cox_hr" in result.columns
        assert "logrank_p" in result.columns
        print("Full pipeline smoke test passed.")
    else:
        print("=== lifelines is NOT installed in this environment ===")
        print("Run: pip install lifelines")
        print("The logrank/derivation/stratification logic above (which needs no")
        print("external package) has been tested and passed. Cox regression,")
        print("the Schoenfeld PH test, and RMST all require lifelines and were")
        print("not exercised in this run.")


if __name__ == "__main__":
    _run_smoke_tests()
