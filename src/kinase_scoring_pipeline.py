"""
Kinase-to-Combination Scoring Pipeline
=======================================

Implements the scoring chain described in Sections 9 and 9a of the TNBC
RTK/NRTK research proposal:

    CTS(k)              -- single-kinase Composite Target Score
    DrugScore(d)         -- per-drug score from its kinase targets
    PairCTS(i, j)        -- RTK-NRTK (or any) kinase-pair score
    DrugPairScore(di,dj) -- drug-pair score (PairCTS + toxicity/synergy)
    TripletCTS(R)        -- 3-drug regimen score

FORMULAS (as confirmed authoritative):

    CTS(k) = w1*Centrality(k) + w2*Essentiality(k) + w3*Survival(k) + w4*Druggability(k)
        w1=0.30, w2=0.25, w3=0.25, w4=0.20   (Section 9 weights)

    PairCTS(i,j) = alpha*CTS(i) + beta*CTS(j) + gamma*Complementarity(i,j) + delta*Crosstalk(i,j)
        (ADDITIVE form -- confirmed authoritative over the multiplicative variant)

    DrugScore(d) = sum(CTS(k) for k in targets(d))          [simple form]
                   refined with potency/selectivity/toxicity (see drug_score_refined)

    DrugPairScore(di,dj) = PairCTS(best target pair) - lambda*Toxicity(di,dj) + mu*Synergy(di,dj)

    TripletCTS(R) = eta*ModuleCoverage(R) + theta*NonRedundancy(R)
                    + kappa*EscapeRouteClosure(R) - rho*CombinedToxicity(R)

IMPORTANT -- NO REAL DATA IS EMBEDDED IN THIS FILE.
All kinase/drug/network values below must come from real sources:
    - Centrality:    NetworkX (betweenness/PageRank/degree) on your STRING /
                      Graphical Lasso / PhosphoSitePlus crosstalk network
    - Essentiality:  DepMap CRISPR (Chronos/CERES) scores
    - Survival:      Cox HR + log-rank p-values from TCGA-BRCA (Section 10)
    - Druggability:  DGIdb + ChEMBL + clinical trial status
    - Drug-target maps: DGIdb / ChEMBL / DrugBank exports
    - Toxicity / synergy: FAERS/CTCAE, GDSC2/PRISM, or your own wet-lab data

The `demo_with_synthetic_data()` function at the bottom generates clearly-
labeled FAKE data purely so you can confirm the pipeline runs end-to-end
before plugging in real inputs. Do not use its output for anything other
than a smoke test.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# 1. NORMALIZATION HELPERS
# =====================================================================

def min_max_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale a raw score column to [0, 1]. Flip direction if lower-is-better
    (e.g. DepMap essentiality scores, where more negative = more essential)."""
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)  # no variance -> neutral score
    scaled = (s - s.min()) / (s.max() - s.min())
    return scaled if higher_is_better else (1.0 - scaled)


# =====================================================================
# 2. CTS(k): SINGLE-KINASE COMPOSITE TARGET SCORE
# =====================================================================

CTS_WEIGHTS = {"centrality": 0.30, "essentiality": 0.25, "survival": 0.25, "druggability": 0.20}


def compute_cts(
    kinase_df: pd.DataFrame,
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Compute CTS(k) for every kinase, keeping ALL kinases in kinase_df's
    index even if some data sources are missing for some of them (rather
    than dropping rows with any missing input).

    MISSING-VALUE POLICY (explicit, confirmed):
        - Missing centrality (betweenness/pagerank/degree all NaN or 0)
          -> centrality term = NEUTRAL (0.5)
        - Missing essentiality (depmap_score NaN)
          -> essentiality term = NEUTRAL (0.5)
        - Missing survival (cox_hr and/or logrank_p NaN)
          -> survival term = NEUTRAL (0.5)
        - Missing druggability (dgidb_score/chembl_count/trial_stage all
          NaN or 0) -> druggability term = LOW (0.0), NOT neutral -- no
          drug-targeting evidence at all is treated as "not currently
          druggable," which is a different (and worse) situation than
          "we don't have data," unlike the other three terms.

    Parameters
    ----------
    kinase_df : DataFrame indexed by kinase_id (should include ALL 90
        kinases even if some columns are NaN for some rows), with columns:
        - 'betweenness', 'pagerank', 'degree'   (network centrality)
        - 'depmap_score'                         (DepMap Chronos/CERES)
        - 'cox_hr', 'logrank_p'                   (survival)
        - 'dgidb_score', 'chembl_count', 'trial_stage'  (druggability)
        Any of these may be NaN for any row; at least one column per
        category must be PRESENT (as a column), even if all-NaN.
    weights : override CTS_WEIGHTS if desired

    Returns
    -------
    DataFrame (same index as kinase_df, all rows retained) with added
    columns: centrality_norm, essentiality_norm, survival_norm,
    druggability_norm, CTS, plus four boolean flag columns
    (centrality_missing, essentiality_missing, survival_missing,
    druggability_missing) so you can audit how many kinases relied on
    a fallback rather than real data.
    """
    w = weights or CTS_WEIGHTS
    df = kinase_df.copy()

    # --- Centrality: average of available normalized network metrics; NEUTRAL fallback ---
    cent_cols = [c for c in ("betweenness", "pagerank", "degree") if c in df.columns]
    if not cent_cols:
        raise ValueError("kinase_df needs at least one of betweenness/pagerank/degree as a COLUMN (values may be NaN)")
    cent_components = [min_max_normalize(df[c], higher_is_better=True) for c in cent_cols]
    centrality_raw = pd.concat(cent_components, axis=1).mean(axis=1)
    # "0 or NA" both count as missing for centrality per the resolution policy --
    # a kinase with literally zero measured centrality is indistinguishable here
    # from one absent from the network, and both fall back to neutral.
    # "0 or NA" for centrality: degree==0 is the reliable signal here -- a node
    # with zero edges is definitionally isolated. PageRank alone is NOT a safe
    # signal for this: it assigns every node a small nonzero "teleportation"
    # baseline value even when fully disconnected (e.g. ~1/N), so requiring
    # ALL THREE metrics to be exactly zero would never trigger for isolated
    # nodes and silently treat their meaningless baseline pagerank as real signal.
    degree_col = "degree" if "degree" in cent_cols else None
    if degree_col:
        all_zero_or_na = df[degree_col].fillna(0).eq(0)
    else:
        all_zero_or_na = df[cent_cols].fillna(0).eq(0).all(axis=1)
    df["centrality_missing"] = centrality_raw.isna() | all_zero_or_na
    df["centrality_norm"] = centrality_raw.where(~df["centrality_missing"], 0.5)

    # --- Essentiality: DepMap score, more negative = more essential; NEUTRAL fallback ---
    if "depmap_score" not in df.columns:
        raise ValueError("kinase_df needs a 'depmap_score' column (values may be NaN)")
    df["essentiality_missing"] = df["depmap_score"].isna()
    essentiality_raw = min_max_normalize(df["depmap_score"], higher_is_better=False)
    df["essentiality_norm"] = essentiality_raw.where(~df["essentiality_missing"], 0.5)

    # --- Survival: combine HR direction with significance; NEUTRAL fallback ---
    if not {"cox_hr", "logrank_p"}.issubset(df.columns):
        raise ValueError("kinase_df needs 'cox_hr' and 'logrank_p' columns (values may be NaN)")
    df["survival_missing"] = df["cox_hr"].isna() | df["logrank_p"].isna()
    p = df["logrank_p"].clip(lower=1e-300)
    raw_survival = np.sign(df["cox_hr"] - 1.0) * (-np.log10(p))
    survival_raw = min_max_normalize(raw_survival, higher_is_better=True)
    df["survival_norm"] = survival_raw.where(~df["survival_missing"], 0.5)

    # --- Druggability: combine DGIdb/ChEMBL/trial evidence; LOW fallback (NOT neutral) ---
    drug_cols = [c for c in ("dgidb_score", "chembl_count", "trial_stage") if c in df.columns]
    if not drug_cols:
        raise ValueError("kinase_df needs at least one of dgidb_score/chembl_count/trial_stage as a COLUMN (values may be NaN)")
    drug_components = [min_max_normalize(df[c], higher_is_better=True) for c in drug_cols]
    druggability_raw = pd.concat(drug_components, axis=1).mean(axis=1)
    all_zero_or_na_drug = df[drug_cols].fillna(0).eq(0).all(axis=1)
    df["druggability_missing"] = druggability_raw.isna() | all_zero_or_na_drug
    df["druggability_norm"] = druggability_raw.where(~df["druggability_missing"], 0.0)  # LOW, not neutral

    # --- Composite (always computable now -- every term has a fallback) ---
    df["CTS"] = (
        w["centrality"] * df["centrality_norm"]
        + w["essentiality"] * df["essentiality_norm"]
        + w["survival"] * df["survival_norm"]
        + w["druggability"] * df["druggability_norm"]
    )
    return df


# =====================================================================
# 3. DRUG -> KINASE TARGET MAPPING & DrugScore(d)
# =====================================================================

def load_drug_target_map(drug_target_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    drug_target_df must have columns: 'drug', 'kinase_id' (one row per drug-target edge).
    Optionally: 'potency', 'selectivity' for the refined DrugScore.

    Example rows:
        afatinib,   EGFR
        afatinib,   ERBB2
        afatinib,   ERBB4
        dasatinib,  SRC
        dasatinib,  ABL1
    """
    mapping: Dict[str, List[str]] = {}
    for drug, group in drug_target_df.groupby("drug"):
        mapping[drug] = group["kinase_id"].tolist()
    return mapping


def drug_score_simple(drug: str, target_map: Dict[str, List[str]], cts: pd.Series) -> float:
    """DrugScore(d) = sum of CTS(k) over the drug's known kinase targets."""
    targets = target_map.get(drug, [])
    return float(sum(cts.get(k, 0.0) for k in targets))


def drug_score_refined(
    drug: str,
    drug_target_df: pd.DataFrame,
    cts: pd.Series,
    off_target_penalty: float = 0.1,
    toxicity_lookup: Optional[Dict[str, float]] = None,
    synergy_prior_lookup: Optional[Dict[str, float]] = None,
) -> float:
    """
    Refined DrugScore weighting potency/selectivity per target, minus an
    off-target burden penalty and single-agent toxicity, plus any known
    synergy prior for this drug (e.g. from prior GDSC2/PRISM screens).

    drug_target_df rows for this drug need 'potency' and 'selectivity' columns
    (both in [0,1], higher = better) in addition to 'kinase_id'.
    """
    rows = drug_target_df[drug_target_df["drug"] == drug]
    if rows.empty:
        return 0.0

    weighted = 0.0
    for _, row in rows.iterrows():
        k = row["kinase_id"]
        potency = row.get("potency", 1.0)
        selectivity = row.get("selectivity", 1.0)
        weighted += cts.get(k, 0.0) * potency * selectivity

    n_targets = len(rows)
    off_target_burden = off_target_penalty * max(0, n_targets - 1)  # >1 target = some off-target risk
    toxicity = (toxicity_lookup or {}).get(drug, 0.0)
    synergy_prior = (synergy_prior_lookup or {}).get(drug, 0.0)

    return weighted - off_target_burden - toxicity + synergy_prior


# =====================================================================
# 4. PairCTS(i,j) & DrugPairScore(di,dj)   [ADDITIVE FORM]
# =====================================================================

PAIR_CTS_WEIGHTS = {"alpha": 0.35, "beta": 0.35, "gamma": 0.20, "delta": 0.10}
DRUG_PAIR_WEIGHTS = {"lambda_toxicity": 0.30, "mu_synergy": 0.30}


def complementarity(
    kinase_i: str,
    kinase_j: str,
    community_map: Dict[str, int],
    reward_cross_community: float = 1.0,
    penalty_same_community: float = 0.0,
) -> float:
    """
    Complementarity term: reward pairs from DIFFERENT network communities
    (Louvain/WGCNA modules) as complementary rather than redundant.
    `community_map`: kinase_id -> community/module id.
    """
    ci = community_map.get(kinase_i)
    cj = community_map.get(kinase_j)
    if ci is None or cj is None:
        return 0.0
    return reward_cross_community if ci != cj else penalty_same_community


def crosstalk_strength(
    kinase_i: str,
    kinase_j: str,
    crosstalk_edges: Dict[Tuple[str, str], float],
) -> float:
    """Look up the RTK-NRTK (or any) crosstalk edge weight (e.g. Graphical
    Lasso partial correlation or STRING combined score, pre-normalized to [0,1]).
    Returns 0 if no edge exists (no known crosstalk)."""
    return crosstalk_edges.get((kinase_i, kinase_j), crosstalk_edges.get((kinase_j, kinase_i), 0.0))


def pair_cts(
    kinase_i: str,
    kinase_j: str,
    cts: pd.Series,
    community_map: Dict[str, int],
    crosstalk_edges: Dict[Tuple[str, str], float],
    weights: Dict[str, float] = None,
) -> float:
    """PairCTS(i,j) = alpha*CTS(i) + beta*CTS(j) + gamma*Complementarity + delta*Crosstalk  [ADDITIVE]"""
    w = weights or PAIR_CTS_WEIGHTS
    comp = complementarity(kinase_i, kinase_j, community_map)
    cross = crosstalk_strength(kinase_i, kinase_j, crosstalk_edges)
    return (
        w["alpha"] * cts.get(kinase_i, 0.0)
        + w["beta"] * cts.get(kinase_j, 0.0)
        + w["gamma"] * comp
        + w["delta"] * cross
    )


def best_target_pair_cts(
    drug_i: str,
    drug_j: str,
    target_map: Dict[str, List[str]],
    cts: pd.Series,
    community_map: Dict[str, int],
    crosstalk_edges: Dict[Tuple[str, str], float],
) -> float:
    """A drug can have multiple targets; take the highest-scoring
    target-pair PairCTS across the two drugs' target sets."""
    targets_i = target_map.get(drug_i, [])
    targets_j = target_map.get(drug_j, [])
    if not targets_i or not targets_j:
        return 0.0
    best = max(
        pair_cts(ki, kj, cts, community_map, crosstalk_edges)
        for ki, kj in itertools.product(targets_i, targets_j)
    )
    return best


def drug_pair_score(
    drug_i: str,
    drug_j: str,
    target_map: Dict[str, List[str]],
    cts: pd.Series,
    community_map: Dict[str, int],
    crosstalk_edges: Dict[Tuple[str, str], float],
    toxicity_lookup: Dict[Tuple[str, str], float],
    synergy_lookup: Dict[Tuple[str, str], float],
    weights: Dict[str, float] = None,
) -> float:
    """DrugPairScore(di,dj) = PairCTS(best targets) - lambda*Toxicity + mu*Synergy"""
    w = weights or DRUG_PAIR_WEIGHTS
    base = best_target_pair_cts(drug_i, drug_j, target_map, cts, community_map, crosstalk_edges)
    tox = toxicity_lookup.get((drug_i, drug_j), toxicity_lookup.get((drug_j, drug_i), 0.0))
    syn = synergy_lookup.get((drug_i, drug_j), synergy_lookup.get((drug_j, drug_i), 0.0))
    return base - w["lambda_toxicity"] * tox + w["mu_synergy"] * syn


# =====================================================================
# 5. TripletCTS(R): 3-DRUG REGIMEN SCORE
# =====================================================================

TRIPLET_WEIGHTS = {"eta": 0.30, "theta": 0.25, "kappa": 0.30, "rho": 0.15}


@dataclass
class Regimen:
    drugs: Tuple[str, str, str]
    targets: List[str] = field(default_factory=list)          # flattened kinase targets hit
    modules_hit: List[int] = field(default_factory=list)       # network community/module ids covered
    escape_routes_closed: List[str] = field(default_factory=list)  # named bypass circuits closed
    combined_toxicity_raw: float = 0.0                          # raw combined toxicity score, any scale


def module_coverage(regimen: Regimen, total_modules: int) -> float:
    """Fraction of known distinct pathway modules the regimen's targets span."""
    if total_modules == 0:
        return 0.0
    return len(set(regimen.modules_hit)) / total_modules


def non_redundancy(regimen: Regimen) -> float:
    """Fraction of the regimen's targets that are NOT duplicated by another
    drug in the same regimen (1.0 = every drug hits a unique node)."""
    if not regimen.targets:
        return 0.0
    unique = len(set(regimen.targets))
    return unique / len(regimen.targets)


def escape_route_closure(regimen: Regimen, total_known_routes: int) -> float:
    """Fraction of documented resistance/bypass escape routes this regimen closes."""
    if total_known_routes == 0:
        return 0.0
    return len(set(regimen.escape_routes_closed)) / total_known_routes


def combined_toxicity_normalized(regimen: Regimen, max_observed_toxicity: float) -> float:
    """Normalize raw combined toxicity to [0,1] against the worst observed
    toxicity across candidate regimens, so rho's penalty is comparable
    regimen-to-regimen."""
    if max_observed_toxicity == 0:
        return 0.0
    return min(regimen.combined_toxicity_raw / max_observed_toxicity, 1.0)


def triplet_cts(
    regimen: Regimen,
    total_modules: int,
    total_known_routes: int,
    max_observed_toxicity: float,
    weights: Dict[str, float] = None,
) -> float:
    """TripletCTS(R) = eta*ModuleCoverage + theta*NonRedundancy + kappa*EscapeRouteClosure - rho*CombinedToxicity"""
    w = weights or TRIPLET_WEIGHTS
    mc = module_coverage(regimen, total_modules)
    nr = non_redundancy(regimen)
    erc = escape_route_closure(regimen, total_known_routes)
    tox = combined_toxicity_normalized(regimen, max_observed_toxicity)
    return w["eta"] * mc + w["theta"] * nr + w["kappa"] * erc - w["rho"] * tox


# =====================================================================
# 6. RANKING UTILITIES
# =====================================================================

def rank_all_pairs(
    drugs: Iterable[str],
    target_map: Dict[str, List[str]],
    cts: pd.Series,
    community_map: Dict[str, int],
    crosstalk_edges: Dict[Tuple[str, str], float],
    toxicity_lookup: Dict[Tuple[str, str], float],
    synergy_lookup: Dict[Tuple[str, str], float],
) -> pd.DataFrame:
    """Score every unique drug pair and return sorted by DrugPairScore descending."""
    rows = []
    for d1, d2 in itertools.combinations(sorted(set(drugs)), 2):
        score = drug_pair_score(
            d1, d2, target_map, cts, community_map, crosstalk_edges, toxicity_lookup, synergy_lookup
        )
        rows.append({"drug_1": d1, "drug_2": d2, "DrugPairScore": score})
    return pd.DataFrame(rows).sort_values("DrugPairScore", ascending=False).reset_index(drop=True)


def rank_all_triplets(
    regimens: List[Regimen],
    total_modules: int,
    total_known_routes: int,
) -> pd.DataFrame:
    """Score a pre-built list of candidate Regimen objects and return sorted
    by TripletCTS descending. Build `regimens` by enumerating drug triplets
    and populating each Regimen's targets/modules_hit/escape_routes_closed/
    combined_toxicity_raw from your real annotation tables first."""
    max_tox = max((r.combined_toxicity_raw for r in regimens), default=0.0)
    rows = []
    for r in regimens:
        score = triplet_cts(r, total_modules, total_known_routes, max_tox)
        rows.append({"drugs": " + ".join(r.drugs), "TripletCTS": score})
    return pd.DataFrame(rows).sort_values("TripletCTS", ascending=False).reset_index(drop=True)


# =====================================================================
# 7. SYNTHETIC SMOKE TEST -- NOT REAL DATA, DO NOT USE FOR ANALYSIS
# =====================================================================

def demo_with_synthetic_data() -> None:
    """Runs the full pipeline on small, clearly-fake data so you can confirm
    everything executes correctly before plugging in real inputs."""
    rng = np.random.default_rng(42)

    kinases = ["EGFR", "ERBB2", "ERBB4", "PIK3CA", "SRC", "ABL1", "JAK1", "JAK2"]
    kinase_df = pd.DataFrame(
        {
            "betweenness": rng.uniform(0, 1, len(kinases)),
            "pagerank": rng.uniform(0, 1, len(kinases)),
            "degree": rng.integers(1, 50, len(kinases)),
            "depmap_score": rng.uniform(-1.5, 0.5, len(kinases)),  # more negative = more essential
            "cox_hr": rng.uniform(0.7, 2.0, len(kinases)),
            "logrank_p": rng.uniform(0.0001, 0.2, len(kinases)),
            "dgidb_score": rng.uniform(0, 1, len(kinases)),
            "chembl_count": rng.integers(0, 200, len(kinases)),
            "trial_stage": rng.integers(0, 4, len(kinases)),
        },
        index=kinases,
    )

    cts_df = compute_cts(kinase_df)
    print("=== CTS (synthetic demo) ===")
    print(cts_df[["CTS"]].sort_values("CTS", ascending=False), "\n")

    drug_target_rows = [
        ("afatinib", "EGFR"), ("afatinib", "ERBB2"), ("afatinib", "ERBB4"),
        ("erlotinib", "EGFR"),
        ("dasatinib", "SRC"), ("dasatinib", "ABL1"),
        ("ruxolitinib", "JAK1"), ("ruxolitinib", "JAK2"),
        ("alpelisib", "PIK3CA"),
        ("trastuzumab", "ERBB2"),
    ]
    drug_target_df = pd.DataFrame(drug_target_rows, columns=["drug", "kinase_id"])
    target_map = load_drug_target_map(drug_target_df)

    print("=== DrugScore (simple, synthetic demo) ===")
    for d in target_map:
        print(f"  {d:12s} {drug_score_simple(d, target_map, cts_df['CTS']):.3f}")
    print()

    community_map = {  # fake module assignment: 0 = RTK/PI3K module, 1 = NRTK module, 2 = JAK module
        "EGFR": 0, "ERBB2": 0, "ERBB4": 0, "PIK3CA": 0,
        "SRC": 1, "ABL1": 1,
        "JAK1": 2, "JAK2": 2,
    }
    crosstalk_edges = {("EGFR", "SRC"): 0.6, ("ERBB2", "PIK3CA"): 0.8, ("SRC", "JAK1"): 0.3}
    toxicity_lookup = {("afatinib", "trastuzumab"): 0.4, ("dasatinib", "ruxolitinib"): 0.2}
    synergy_lookup = {("afatinib", "alpelisib"): 0.5, ("afatinib", "trastuzumab"): 0.3}

    print("=== Ranked drug pairs (synthetic demo) ===")
    pairs_df = rank_all_pairs(
        target_map.keys(), target_map, cts_df["CTS"], community_map,
        crosstalk_edges, toxicity_lookup, synergy_lookup,
    )
    print(pairs_df.head(10), "\n")

    # Build a couple of candidate triplets manually for the demo
    regimens = [
        Regimen(
            drugs=("afatinib", "alpelisib", "trastuzumab"),
            targets=["EGFR", "PIK3CA", "ERBB2"],
            modules_hit=[0, 0, 0],
            escape_routes_closed=["EGFR_bypass_via_ERBB2", "PI3K_reactivation"],
            combined_toxicity_raw=0.5,
        ),
        Regimen(
            drugs=("dasatinib", "ruxolitinib", "alpelisib"),
            targets=["SRC", "JAK1", "PIK3CA"],
            modules_hit=[1, 2, 0],
            escape_routes_closed=["SRC_JAK_crosstalk_bypass"],
            combined_toxicity_raw=0.7,
        ),
    ]
    print("=== Ranked triplets (synthetic demo) ===")
    triplets_df = rank_all_triplets(regimens, total_modules=3, total_known_routes=3)
    print(triplets_df, "\n")

    print("Demo complete. Replace every synthetic input above with real")
    print("TCGA-BRCA / DepMap / STRING / DGIdb / GDSC2-PRISM data before")
    print("drawing any biological conclusions.")


if __name__ == "__main__":
    demo_with_synthetic_data()
