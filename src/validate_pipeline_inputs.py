"""
Pipeline Input Validator
========================

Run this BEFORE the real batch run (the script importing compute_cts,
rank_all_pairs, rank_all_triplets, etc.). It checks every input file for:
    - existence and parseability
    - required columns present (and flags column-NAME MISMATCHES against
      what the downstream functions actually expect -- e.g. DGIdb's
      'gene'/'source' vs fetch_dgidb_interactions()'s 'kinase_id'/'sources')
    - orientation bugs (e.g. DepMap rows vs columns)
    - value ranges that are physically/statistically impossible or
      suspiciously unnormalized (logrank_p > 1, STRING weights on a
      0-1000 scale instead of [0,1], etc.)
    - coverage against your kinase_list / drug_list (e.g. "38 of 90
      kinases have no community assignment")

Every check produces a PASS / WARN / FAIL. FAILs mean the downstream
pipeline WILL produce wrong numbers (or crash) if you proceed -- fix
these first. WARNs are things to eyeball before trusting the output.

Usage:
    python3 validate_pipeline_inputs.py
(edit the PATHS section below to match your real batch script's paths,
or import validate_all() and call it with your own path dict)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =====================================================================
# RESULT TYPE + REPORT PRINTER
# =====================================================================

@dataclass
class CheckResult:
    level: str  # "PASS", "WARN", "FAIL"
    check: str
    message: str


def _p(level: str, check: str, message: str) -> CheckResult:
    return CheckResult(level, check, message)


def print_report(results: List[CheckResult]) -> bool:
    """Print a grouped report. Returns True if there are zero FAILs."""
    icons = {"PASS": "\u2705", "WARN": "\u26a0\ufe0f ", "FAIL": "\u274c"}
    n_fail = sum(r.level == "FAIL" for r in results)
    n_warn = sum(r.level == "WARN" for r in results)
    n_pass = sum(r.level == "PASS" for r in results)

    for r in results:
        print(f"  {icons[r.level]} [{r.level:4s}] {r.check}: {r.message}")

    print(f"\n{'='*70}")
    print(f"  {n_pass} passed, {n_warn} warnings, {n_fail} failures")
    print(f"{'='*70}")
    if n_fail:
        print("  FAIL means the real batch run WILL produce wrong or missing")
        print("  numbers if you proceed without fixing these first.")
    elif n_warn:
        print("  No blocking failures, but review the warnings above --")
        print("  they won't crash the pipeline, but may silently skew scores.")
    else:
        print("  All checks passed. Safe to proceed to the full batch run.")
    return n_fail == 0


# =====================================================================
# GENERIC HELPERS
# =====================================================================

def _check_file_exists(path: Path, label: str) -> Optional[CheckResult]:
    if not path.exists():
        return _p("FAIL", label, f"File not found: {path}")
    if path.stat().st_size == 0:
        return _p("FAIL", label, f"File exists but is empty: {path}")
    return None


def _check_columns(df: pd.DataFrame, required: List[str], label: str) -> List[CheckResult]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [_p("FAIL", label, f"Missing required column(s) {missing}. Found: {list(df.columns)}")]
    return [_p("PASS", label, f"All required columns present: {required}")]


def _check_range(series: pd.Series, lo: float, hi: float, label: str, allow_na: bool = True) -> CheckResult:
    s = pd.to_numeric(series, errors="coerce")
    n_na = s.isna().sum()
    if not allow_na and n_na > 0:
        return _p("WARN", label, f"{n_na} non-numeric/missing values found")
    s_valid = s.dropna()
    if s_valid.empty:
        return _p("FAIL", label, "No valid numeric values to check")
    out_of_range = ((s_valid < lo) | (s_valid > hi)).sum()
    if out_of_range > 0:
        return _p(
            "FAIL", label,
            f"{out_of_range}/{len(s_valid)} values outside expected range [{lo}, {hi}] "
            f"(actual range: [{s_valid.min():.4g}, {s_valid.max():.4g}])",
        )
    return _p("PASS", label, f"All values within [{lo}, {hi}] (actual range: [{s_valid.min():.4g}, {s_valid.max():.4g}])")


def _check_coverage(items: List[str], available: set, label: str, min_coverage: float = 0.5) -> CheckResult:
    if not items:
        return _p("FAIL", label, "Item list is empty")
    found = [i for i in items if i in available]
    coverage = len(found) / len(items)
    missing_examples = [i for i in items if i not in available][:5]
    if coverage == 0:
        return _p("FAIL", label, f"0/{len(items)} items found -- check for naming/ID-format mismatch (e.g. 'EGFR' vs 'EGFR (1956)')")
    if coverage < min_coverage:
        return _p("FAIL", label, f"Only {len(found)}/{len(items)} ({coverage:.0%}) found, below the {min_coverage:.0%} threshold. Missing e.g.: {missing_examples}")
    if coverage < 1.0:
        return _p("WARN", label, f"{len(found)}/{len(items)} ({coverage:.0%}) found. Missing e.g.: {missing_examples}")
    return _p("PASS", label, f"All {len(items)} items found")


# =====================================================================
# 1. DEPMAP ESSENTIALITY FILE
# =====================================================================

def validate_depmap_processed(path: Path, kinase_list: List[str]) -> List[CheckResult]:
    """
    Validates the TIDY, already-processed DepMap output (as written by
    parse_depmap_gene_effect() + a groupby/mean step), NOT the raw wide
    CRISPRGeneEffect.csv file -- use validate_depmap() instead if you're
    checking the raw file directly. Expected columns: kinase_id, mean_depmap_score.
    """
    label = "DepMap (processed)"
    err = _check_file_exists(path, label)
    if err:
        return [err]

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        return [_p("FAIL", label, f"Could not parse file: {exc}")]

    results = _check_columns(df, ["kinase_id", "mean_depmap_score"], label)
    if any(r.level == "FAIL" for r in results):
        return results

    results.append(_check_range(df["mean_depmap_score"], lo=-10, hi=10, label=f"{label}.mean_depmap_score"))
    results.append(_check_coverage(kinase_list, set(df["kinase_id"]), f"{label}.coverage"))

    dupes = df["kinase_id"][df["kinase_id"].duplicated()].unique().tolist()
    if dupes:
        results.append(_p("WARN", label, f"{len(dupes)} duplicate kinase_id rows found (e.g. {dupes[:5]})"))

    return results


def validate_depmap(path: Path, kinase_list: List[str]) -> List[CheckResult]:
    label = "DepMap"
    results = []
    err = _check_file_exists(path, label)
    if err:
        return [err]

    try:
        df = pd.read_csv(path, sep=None, engine="python", nrows=5)
    except Exception as exc:
        return [_p("FAIL", label, f"Could not parse file: {exc}")]

    df_full_cols = pd.read_csv(path, sep=None, engine="python", nrows=0).columns.tolist()

    # Orientation check: real DepMap gene-effect files have genes as COLUMNS
    # named like 'EGFR (1956)', and cell lines as the first-column ROW index
    # (ModelID, e.g. 'ACH-000001', or a cell line name).
    stripped_cols = {c.split(" (")[0] for c in df_full_cols}
    kinase_overlap_in_columns = len(stripped_cols & set(kinase_list))

    first_col_sample = pd.read_csv(path, sep=None, engine="python", usecols=[0], nrows=20).iloc[:, 0].astype(str).tolist()
    kinase_overlap_in_first_column = sum(1 for v in first_col_sample if v in kinase_list)

    if kinase_overlap_in_columns == 0 and kinase_overlap_in_first_column > 0:
        results.append(_p(
            "FAIL", label,
            f"Orientation looks BACKWARDS: {kinase_overlap_in_first_column} kinase symbols found in the "
            f"first column (rows), 0 found in the column headers. DepMap gene-effect files should have "
            f"genes as COLUMNS ('EGFR (1956)') and cell lines as ROWS. If you're filtering by cell-line "
            f"substrings on the wrong axis (e.g. `depmap_df[tnbc_lines].mean(axis=1)` where tnbc_lines "
            f"was matched against columns), you are averaging over the wrong axis.",
        ))
    elif kinase_overlap_in_columns == 0:
        results.append(_p(
            "FAIL", label,
            f"None of your {len(kinase_list)} kinases found in column headers (checked both raw and "
            f"'(ID)'-stripped forms). First 5 columns: {df_full_cols[:5]}",
        ))
    else:
        coverage = kinase_overlap_in_columns / len(kinase_list)
        level = "PASS" if coverage == 1.0 else "WARN"
        results.append(_p(level, label, f"{kinase_overlap_in_columns}/{len(kinase_list)} kinases found in column headers (orientation looks correct)"))

    # Value range: Chronos/CERES scores are typically roughly in [-2, 1.5].
    # Values wildly outside this (e.g. all in the hundreds/thousands, or all
    # exactly 0) suggest the wrong file or a unit/parsing problem.
    try:
        sample_vals = pd.read_csv(path, sep=None, engine="python", nrows=50).select_dtypes(include=[np.number])
        if not sample_vals.empty:
            flat = sample_vals.values.flatten()
            flat = flat[~np.isnan(flat)]
            if len(flat) > 0:
                if np.all(flat == 0):
                    results.append(_p("FAIL", label, "All sampled values are exactly 0 -- likely a parsing/orientation error"))
                elif flat.min() < -10 or flat.max() > 10:
                    results.append(_p("WARN", label, f"Sampled value range [{flat.min():.2f}, {flat.max():.2f}] is unusually wide for Chronos/CERES scores (typically ~[-2, 1.5]) -- confirm this is gene-effect data, not raw read counts"))
                else:
                    results.append(_p("PASS", label, f"Sampled value range [{flat.min():.2f}, {flat.max():.2f}] looks consistent with Chronos/CERES scores"))
    except Exception as exc:
        results.append(_p("WARN", label, f"Could not sample values for range check: {exc}"))

    return results


# =====================================================================
# 2. TCGA SURVIVAL STATS FILE
# =====================================================================

def validate_survival(path: Path, kinase_list: List[str]) -> List[CheckResult]:
    label = "Survival"
    err = _check_file_exists(path, label)
    if err:
        return [err]

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        return [_p("FAIL", label, f"Could not parse file: {exc}")]

    results = _check_columns(df, ["kinase_id", "cox_hr", "logrank_p"], label)
    if any(r.level == "FAIL" for r in results):
        return results

    df = df.set_index("kinase_id")
    results.append(_check_range(df["cox_hr"], lo=0.001, hi=100, label=f"{label}.cox_hr"))
    results.append(_check_range(df["logrank_p"], lo=0.0, hi=1.0, label=f"{label}.logrank_p"))
    results.append(_check_coverage(kinase_list, set(df.index), f"{label}.coverage"))

    dupes = df.index[df.index.duplicated()].unique().tolist()
    if dupes:
        results.append(_p("WARN", label, f"{len(dupes)} duplicate kinase_id rows found (e.g. {dupes[:5]}) -- compute_cts() expects one row per kinase"))

    return results


# =====================================================================
# 3. DGIDB DRUG-TARGET FILE
# =====================================================================

def validate_dgidb(path: Path) -> List[CheckResult]:
    label = "DGIdb"
    err = _check_file_exists(path, label)
    if err:
        return [err]

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        return [_p("FAIL", label, f"Could not parse file: {exc}")]

    results = []
    fetcher_cols = {"kinase_id", "drug", "interaction_score", "interaction_types", "sources"}
    batch_script_cols = {"gene", "drug", "source", "interaction_type"}
    found_cols = set(df.columns)

    if fetcher_cols.issubset(found_cols):
        results.append(_p("PASS", label, "Columns match fetch_dgidb_interactions()'s output format directly (kinase_id/sources/interaction_types)"))
    elif batch_script_cols.issubset(found_cols):
        results.append(_p(
            "WARN", label,
            "Columns match the OLD/batch-script naming (gene/source/interaction_type), not "
            "fetch_dgidb_interactions()'s actual output (kinase_id/sources/interaction_types). "
            "The batch script's `dgidb_df.rename(columns={'gene': 'kinase_id'})` step is required "
            "and column 'source'/'interaction_type' will double-count as near-duplicate druggability "
            "signals -- see the review notes on dgidb_score vs chembl_count.",
        ))
    else:
        results.append(_p("FAIL", label, f"Columns match neither expected format. Found: {sorted(found_cols)}"))

    if "kinase_id" in found_cols or "gene" in found_cols:
        gene_col = "kinase_id" if "kinase_id" in found_cols else "gene"
        n_null_drug = df["drug"].isna().sum() if "drug" in found_cols else None
        if n_null_drug:
            results.append(_p("WARN", label, f"{n_null_drug} rows have a null drug name"))
        results.append(_p("PASS", label, f"{df[gene_col].nunique()} unique genes, {len(df)} total interaction rows"))

    return results


# =====================================================================
# 4. STRING CROSSTALK EDGES FILE
# =====================================================================

def validate_string_edges(path: Path, kinase_list: List[str]) -> List[CheckResult]:
    label = "STRING edges"
    err = _check_file_exists(path, label)
    if err:
        return [err]

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        return [_p("FAIL", label, f"Could not parse file: {exc}")]

    results = _check_columns(df, ["source", "target", "weight"], label)
    if any(r.level == "FAIL" for r in results):
        return results

    w = pd.to_numeric(df["weight"], errors="coerce").dropna()
    if w.empty:
        results.append(_p("FAIL", label, "No valid numeric weight values"))
    elif w.max() > 1.5:
        results.append(_p(
            "FAIL", label,
            f"Weight range [{w.min():.2f}, {w.max():.2f}] looks like RAW STRING combined_scores "
            "(typically 0-1000), not normalized to [0,1]. pair_cts()'s additive formula assumes "
            "crosstalk_strength is pre-scaled to [0,1] -- divide by 1000 (or the actual max) before use, "
            "or this term will dominate PairCTS and drown out CTS(i)/CTS(j).",
        ))
    else:
        results.append(_p("PASS", label, f"Weight range [{w.min():.3f}, {w.max():.3f}] looks normalized"))

    nodes = set(df["source"]) | set(df["target"])
    results.append(_check_coverage(kinase_list, nodes, f"{label}.node_coverage"))

    return results


# =====================================================================
# 5. COMMUNITY / MODULE MAP FILE
# =====================================================================

def validate_community_map(path: Path, kinase_list: List[str]) -> List[CheckResult]:
    label = "Community map"
    err = _check_file_exists(path, label)
    if err:
        return [err]

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        return [_p("FAIL", label, f"Could not parse file: {exc}")]

    results = _check_columns(df, ["kinase_id", "community"], label)
    if any(r.level == "FAIL" for r in results):
        return results

    results.append(_check_coverage(kinase_list, set(df["kinase_id"]), f"{label}.coverage"))

    n_communities = df["community"].nunique()
    if n_communities < 2:
        results.append(_p("FAIL", label, f"Only {n_communities} distinct community found -- complementarity() rewards CROSS-community pairs, so with <2 communities every pair looks 'redundant'"))
    else:
        results.append(_p("PASS", label, f"{n_communities} distinct communities found"))

    return results


# =====================================================================
# 6. DRUG LIST / KINASE LIST FILES
# =====================================================================

def validate_list_file(path: Path, label: str, expected_min: int = 1) -> List[CheckResult]:
    err = _check_file_exists(path, label)
    if err:
        return [err]

    items = [line.strip() for line in open(path) if line.strip()]
    results = []
    if len(items) < expected_min:
        results.append(_p("FAIL", label, f"Only {len(items)} entries, expected at least {expected_min}"))
    else:
        results.append(_p("PASS", label, f"{len(items)} entries"))

    dupes = [item for item, count in pd.Series(items).value_counts().items() if count > 1]
    if dupes:
        results.append(_p("WARN", label, f"{len(dupes)} duplicate entries found (e.g. {dupes[:5]})"))

    # Allow single spaces between words (real drug names can be two words,
    # e.g. "cofetuzumab pelidotin") -- but still catch genuine formatting
    # problems: tabs, commas, leading/trailing whitespace, or doubled spaces.
    blank_like = [i for i in items if not re.match(r"^[A-Za-z0-9_.\-]+( [A-Za-z0-9_.\-]+)*$", i)]
    if blank_like:
        results.append(_p("WARN", label, f"{len(blank_like)} entries contain unexpected characters/whitespace (e.g. {blank_like[:3]}) -- check for stray commas, tabs, or header rows"))

    return results


# =====================================================================
# 7. FAERS TOXICITY DIRECTORY
# =====================================================================

def validate_faers(dir_path: Path, drug_list: List[str]) -> List[CheckResult]:
    label = "FAERS"
    if not dir_path.exists():
        return [_p("FAIL", label, f"Directory not found: {dir_path}")]

    found = 0
    bad_format = []
    for drug in drug_list:
        f = dir_path / f"{drug}_faers.tsv"
        if not f.exists():
            continue
        found += 1
        try:
            df = pd.read_csv(f, sep="\t", nrows=5)
            if not {"reaction", "report_count"}.issubset(df.columns):
                bad_format.append(drug)
        except Exception:
            bad_format.append(drug)

    results = []
    coverage = found / len(drug_list) if drug_list else 0
    level = "PASS" if coverage == 1.0 else ("WARN" if coverage >= 0.5 else "FAIL")
    results.append(_p(level, label, f"{found}/{len(drug_list)} drugs have a FAERS file (missing drugs silently get toxicity=0.0 in the batch script, which understates their risk rather than leaving it unknown)"))

    if bad_format:
        results.append(_p("FAIL", label, f"{len(bad_format)} FAERS files have unexpected columns (expected 'reaction','report_count'): {bad_format[:5]}"))

    return results


# =====================================================================
# RUN EVERYTHING
# =====================================================================

def validate_all(paths: Dict[str, Path], drug_list: List[str], kinase_list: List[str]) -> bool:
    all_results: List[CheckResult] = []

    print("Validating DepMap essentiality file...")
    all_results += validate_depmap_processed(paths["depmap"], kinase_list)

    print("Validating TCGA survival stats file...")
    all_results += validate_survival(paths["survival"], kinase_list)

    print("Validating DGIdb file...")
    all_results += validate_dgidb(paths["dgidb"])

    print("Validating STRING edges file...")
    all_results += validate_string_edges(paths["string_edges"], kinase_list)

    print("Validating community map file...")
    all_results += validate_community_map(paths["community"], kinase_list)

    print("Validating drug list file...")
    all_results += validate_list_file(paths["drug_list"], "Drug list", expected_min=2)

    print("Validating kinase list file...")
    all_results += validate_list_file(paths["kinase_list"], "Kinase list", expected_min=2)

    print("Validating FAERS directory...")
    all_results += validate_faers(paths["faers_dir"], drug_list)

    print()
    return print_report(all_results)


if __name__ == "__main__":
    from pathlib import Path

    BASE = Path.home() / "rtk_nrtk_tnbc"
    PATHS = {
        "depmap": BASE / "data/processed/depmap/depmap_tnbc_essentiality.tsv",
        "survival": BASE / "data/processed/tcga_brca/survival_stats.tsv",
        "dgidb": BASE / "data/processed/dgidb/dgidb_interactions.tsv",
        "string_edges": BASE / "data/processed/string/string_edges.tsv",
        "community": BASE / "data/processed/string/community_map.tsv",
        "drug_list": BASE / "data/raw/drugs/drug_list.txt",
        "kinase_list": BASE / "data/raw/kinases/kinase_90_list.txt",
        "faers_dir": BASE / "data/processed/faers",
    }
    kinase_list = [k.strip() for k in open(PATHS["kinase_list"])] if PATHS["kinase_list"].exists() else []
    drug_list = [d.strip() for d in open(PATHS["drug_list"])] if PATHS["drug_list"].exists() else []

    ok = validate_all(PATHS, drug_list, kinase_list)
    exit(0 if ok else 1)
