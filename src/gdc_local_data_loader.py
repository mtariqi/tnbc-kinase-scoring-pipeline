"""
GDC Local Data Loader
=====================

Reads a locally-downloaded GDCdata/TCGA-BRCA tree (the standard layout
TCGAbiolinks::GDCdownload() creates) into the expression matrix and
mutation table the rest of this pipeline (kinase_scoring_pipeline.py,
tcga_brca_survival_pipeline.py) expects.

EXPECTED LAYOUT (TCGAbiolinks default):
    <root>/GDCdata/TCGA-BRCA/harmonized/
        Transcriptome_Profiling/Gene_Expression_Quantification/<file-uuid>/<sample>.rna_seq.augmented_star_gene_counts.tsv
        Simple_Nucleotide_Variation/Masked_Somatic_Mutation/<file-uuid>/<sample>.wxs...maf(.gz)

    <root>/gdc_sample_sheet*.tsv   <- REQUIRED to map file UUIDs to patient/case barcodes.
        This is the file GDC's web portal (or TCGAbiolinks' getResults()/
        GDCquery object) generates alongside the download. Without it,
        file-UUID folder names cannot be reliably linked to patients.
        If you don't have one, re-derive it from your original GDCquery
        object in R via `getResults(query)` and export it to TSV.

STEP 0 -- RUN THIS FIRST ON YOUR REAL DATA:
    inspect_gdc_directory("/path/to/rtk_nrtk_tnbc/data/raw/tcga_brca")
This prints what file types/extensions actually exist and a few example
paths, so you (or I, if you paste the output back) can confirm this
loader's assumptions match your actual download before trusting any
parsed numbers.

This module was tested against a synthetic directory tree built with the
real GDC STAR-counts TSV header format and real MAF column layout (see
_run_smoke_tests at the bottom) -- not against your actual 21.4GB dataset,
which I don't have access to. Run inspect_gdc_directory() on your real
path and compare its output to the smoke-test structure before trusting
build_expression_matrix()/load_combined_maf() on your real files.
"""

from __future__ import annotations

import gzip
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


# =====================================================================
# STEP 0: STRUCTURE INSPECTOR -- RUN THIS FIRST ON YOUR REAL PATH
# =====================================================================

def inspect_gdc_directory(root_path: str, max_examples_per_ext: int = 3) -> None:
    """
    Walk root_path and print a summary: how many files of each extension/
    naming pattern exist, and a few example full paths for each. Prints
    directly (rather than returning a DataFrame) so it's easy to eyeball
    and paste into a chat message if you want a second opinion on it.
    """
    root = Path(root_path)
    if not root.exists():
        print(f"Path does not exist: {root_path}")
        return

    ext_counter: Counter = Counter()
    examples: Dict[str, List[str]] = {}
    top_level_dirs = set()

    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) >= 1:
            top_level_dirs.add(rel.parts[0])
        for fname in filenames:
            # Use a naming-pattern key, not just the raw extension, since
            # GDC files often have multi-part suffixes like .rna_seq.augmented_star_gene_counts.tsv
            key = _classify_filename(fname)
            ext_counter[key] += 1
            if len(examples.get(key, [])) < max_examples_per_ext:
                examples.setdefault(key, []).append(str(Path(dirpath) / fname))

    print(f"=== Top-level entries under {root_path} ===")
    for d in sorted(top_level_dirs):
        print(f"  {d}")

    print(f"\n=== File types found (n={sum(ext_counter.values())} files total) ===")
    for key, count in ext_counter.most_common():
        print(f"  {count:6d}  x  {key}")
        for ex in examples[key]:
            print(f"           e.g. {ex}")

    sheet_candidates = sorted(set(root.glob("gdc_sample_sheet*.tsv")) | set(root.glob("**/gdc_sample_sheet*.tsv")))
    print(f"\n=== gdc_sample_sheet*.tsv found: {len(sheet_candidates)} ===")
    for s in sheet_candidates:
        print(f"  {s}")
    if not sheet_candidates:
        print("  None found -- you'll need this (or an equivalent file-UUID -> "
              "case/sample-barcode map) to attach patient identities to the files below.")


def _classify_filename(fname: str) -> str:
    """Collapse a real filename to a reusable pattern key, e.g.
    'sample1.rna_seq.augmented_star_gene_counts.tsv' -> 'rna_seq.augmented_star_gene_counts.tsv'"""
    parts = fname.split(".")
    if len(parts) <= 2:
        return fname
    return ".".join(parts[1:])  # drop the sample-specific first token


# =====================================================================
# 1. SAMPLE SHEET -- FILE UUID -> PATIENT/CASE BARCODE MAP
# =====================================================================

def load_sample_sheet(sample_sheet_path: str) -> pd.DataFrame:
    """
    Load a GDC sample sheet TSV. Expected columns (GDC's standard export):
        'File ID', 'File Name', 'Data Category', 'Data Type',
        'Project ID', 'Case ID', 'Sample ID', 'Sample Type'
    """
    df = pd.read_csv(sample_sheet_path, sep="\t")
    required = {"File ID", "File Name", "Case ID", "Sample ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Sample sheet is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


# =====================================================================
# 2. EXPRESSION: GDC STAR-COUNTS FILES
# =====================================================================

def find_expression_files(root_path: str) -> List[str]:
    """Locate all *augmented_star_gene_counts.tsv files under root_path."""
    root = Path(root_path)
    return [str(p) for p in root.rglob("*augmented_star_gene_counts.tsv")]


def load_star_counts_file(path: str, value_col: str = "tpm_unstranded") -> pd.Series:
    """
    Parse one GDC STAR gene-counts TSV into a gene_name -> value Series.

    Real GDC STAR-counts files have 2 leading '#' comment lines, then a
    header row, then 4 summary rows (N_unmapped/N_multimapping/N_noFeature/
    N_ambiguous) before the actual per-gene rows -- both are handled here.

    value_col options (all present in the real file): 'unstranded',
    'stranded_first', 'stranded_second', 'tpm_unstranded', 'fpkm_unstranded',
    'fpkm_uq_unstranded'. Use 'tpm_unstranded' for cross-sample comparability.
    """
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[~df["gene_id"].str.startswith("N_")]  # drop the 4 summary rows
    df = df.dropna(subset=[value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    # A gene_name can appear more than once (different Ensembl IDs); keep
    # the max value per symbol rather than silently overwriting.
    return df.groupby("gene_name")[value_col].max()


def build_expression_matrix(
    root_path: str,
    sample_sheet_path: str,
    gene_symbols: Optional[List[str]] = None,
    value_col: str = "tpm_unstranded",
) -> pd.DataFrame:
    """
    Build a patient (Case ID) x gene expression matrix from every STAR-counts
    file under root_path, restricted to `gene_symbols` if given (pass your
    90 RTK/NRTK kinases here to avoid loading unnecessary columns).

    Returns a DataFrame indexed by Case ID (e.g. 'TCGA-AO-A128').
    If a case has multiple samples/files, they are averaged.
    """
    sheet = load_sample_sheet(sample_sheet_path)
    file_to_case = dict(zip(sheet["File Name"], sheet["Case ID"]))

    files = find_expression_files(root_path)
    if not files:
        raise FileNotFoundError(
            f"No *augmented_star_gene_counts.tsv files found under {root_path}. "
            "Run inspect_gdc_directory() to see what's actually there -- your "
            "GDC data may use a different file-naming convention (e.g. legacy "
            "htseq.counts.gz), which this function does not yet parse."
        )

    rows = {}
    unmapped_files = []
    for fpath in files:
        fname = Path(fpath).name
        case_id = file_to_case.get(fname)
        if case_id is None:
            unmapped_files.append(fname)
            continue
        series = load_star_counts_file(fpath, value_col=value_col)
        if gene_symbols:
            series = series.reindex(gene_symbols)
        rows[case_id] = series

    if unmapped_files:
        print(f"Warning: {len(unmapped_files)} expression files had no entry in the "
              f"sample sheet and were skipped: {unmapped_files[:5]}{'...' if len(unmapped_files) > 5 else ''}")

    matrix = pd.DataFrame(rows).T
    matrix.index.name = "patient_id"
    # If a case had multiple files, average across them.
    return matrix.groupby(matrix.index).mean()


# =====================================================================
# 3. MUTATIONS: MAF FILES
# =====================================================================

def find_maf_files(root_path: str) -> List[str]:
    """Locate all Masked Somatic Mutation MAF files (plain or .gz) under root_path."""
    root = Path(root_path)
    return [str(p) for p in root.rglob("*.maf")] + [str(p) for p in root.rglob("*.maf.gz")]


def _open_maybe_gz(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_one_maf(path: str) -> pd.DataFrame:
    """Parse one MAF file, skipping its leading '#version...' comment line(s)."""
    with _open_maybe_gz(path) as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines[header_idx:])), sep="\t")


def load_combined_maf(
    root_path: str,
    gene_symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Concatenate every MAF file under root_path into one long mutation table,
    optionally filtered to `gene_symbols` (your 90 kinases). Returns columns
    at minimum: Hugo_Symbol, Variant_Classification, Variant_Type,
    Tumor_Sample_Barcode (plus whatever else each MAF carries).
    """
    files = find_maf_files(root_path)
    if not files:
        raise FileNotFoundError(
            f"No .maf/.maf.gz files found under {root_path}. Run "
            "inspect_gdc_directory() to check the actual file-naming pattern."
        )

    frames = [load_one_maf(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    if gene_symbols:
        combined = combined[combined["Hugo_Symbol"].isin(gene_symbols)]
    return combined


def maf_to_patient_gene_flags(maf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a long mutation table into a patient x gene binary matrix
    (1 = at least one mutation of any type in that gene for that patient).
    Uses the first 12 characters of Tumor_Sample_Barcode as the patient id
    (TCGA barcode convention, e.g. 'TCGA-AO-A128-01A' -> 'TCGA-AO-A128').
    """
    df = maf_df.copy()
    df["patient_id"] = df["Tumor_Sample_Barcode"].str[:12]
    flags = (
        df.groupby(["patient_id", "Hugo_Symbol"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )
    return flags


# =====================================================================
# 4. SMOKE TESTS -- AGAINST A SYNTHETIC BUT FORMAT-ACCURATE DIRECTORY
# =====================================================================

def _run_smoke_tests(test_root: str) -> None:
    print("=== inspect_gdc_directory() on the synthetic test tree ===")
    inspect_gdc_directory(test_root)

    sample_sheet_path = next(str(p) for p in Path(test_root).glob("gdc_sample_sheet*.tsv"))

    print("\n=== load_sample_sheet() ===")
    sheet = load_sample_sheet(sample_sheet_path)
    print(sheet, "\n")
    assert len(sheet) == 3

    print("=== load_star_counts_file() on one file ===")
    files = find_expression_files(test_root)
    assert len(files) == 2, f"expected 2 expression files, found {len(files)}"
    series = load_star_counts_file(files[0])
    print(series, "\n")
    assert "EGFR" in series.index
    assert series["EGFR"] > 0

    print("=== build_expression_matrix() across both files ===")
    expr_matrix = build_expression_matrix(test_root, sample_sheet_path, gene_symbols=["EGFR", "ERBB2", "SRC"])
    print(expr_matrix, "\n")
    assert expr_matrix.shape == (2, 3)
    assert set(expr_matrix.index) == {"TCGA-AO-A128", "TCGA-A8-A08B"}
    assert abs(expr_matrix.loc["TCGA-AO-A128", "EGFR"] - 120.5) < 1e-6

    print("=== find_maf_files() + load_combined_maf() ===")
    maf_files = find_maf_files(test_root)
    print(f"  found {len(maf_files)} MAF file(s): {maf_files}")
    assert len(maf_files) == 2  # both the .maf and .maf.gz copies in this test tree

    combined = load_combined_maf(test_root, gene_symbols=["EGFR", "PTEN", "TP53"])
    print(combined, "\n")
    assert set(combined["Hugo_Symbol"]) == {"EGFR", "PTEN", "TP53"}

    print("=== maf_to_patient_gene_flags() ===")
    flags = maf_to_patient_gene_flags(combined)
    print(flags, "\n")
    assert flags.loc["TCGA-AO-A128", "EGFR"] == 1
    assert flags.loc["TCGA-AO-A128", "TP53"] == 1

    print("All smoke tests passed against the synthetic (format-accurate) test tree.")
    print("Next: run inspect_gdc_directory() on your REAL 21.4GB path and compare")
    print("the output to what you see above before trusting the parsed numbers.")


# =====================================================================
# 5. ALTERNATIVE PATH: LOAD FROM R/TCGAbiolinks-EXPORTED CSVs
#    (use this instead of the raw-folder-walking functions above if you
#    ran assemble_tcga_brca_data.R -- it's simpler since R/GDCprepare()
#    already did the file-parsing and patient-barcode matching for you)
# =====================================================================

def load_r_exported_expression(csv_path: str, gene_symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load expression_tpm.csv as written by assemble_tcga_brca_data.R:
    rows = samples (full TCGA barcode as the index), one 'patient_id'
    column (first 12 chars of the barcode), remaining columns = genes.
    Returns a patient_id x gene matrix, averaged if a patient has >1 sample.
    """
    df = pd.read_csv(csv_path, index_col=0)
    if "patient_id" not in df.columns:
        raise ValueError("Expected a 'patient_id' column -- was this file written by assemble_tcga_brca_data.R?")
    patient_col = df.pop("patient_id")
    if gene_symbols:
        missing = set(gene_symbols) - set(df.columns)
        if missing:
            print(f"Warning: {len(missing)} requested genes not found in expression_tpm.csv: {sorted(missing)[:10]}")
        df = df[[g for g in gene_symbols if g in df.columns]]
    df["patient_id"] = patient_col
    return df.groupby("patient_id").mean()


def load_r_exported_clinical(csv_path: str) -> pd.DataFrame:
    """
    Load clinical.csv as written by GDCquery_clinic() + assemble_tcga_brca_data.R.
    Renames GDC's clinical column names to the ones
    tcga_brca_survival_pipeline.derive_survival_time_event() expects, where
    the mapping is unambiguous. Prints a warning for anything it can't map
    automatically -- check these manually before running the survival pipeline.
    """
    df = pd.read_csv(csv_path)

    rename_candidates = {
        "days_to_death": "days_to_death",
        "days_to_last_follow_up": "days_to_last_follow_up",
        "vital_status": "vital_status",
        "ajcc_pathologic_stage": "stage_raw",
    }
    present = {k: v for k, v in rename_candidates.items() if k in df.columns}
    df = df.rename(columns=present)

    # Age needs an explicit either/or choice -- renaming both age_at_index AND
    # age_at_diagnosis to "age" (if both are present, which is common) creates
    # two columns with the same name. Pandas allows this silently, and every
    # downstream Cox regression then fails identically (merged['age'] returns
    # a DataFrame, not a Series) with no clear error pointing back to this.
    if "age_at_index" in df.columns:
        df["age"] = df["age_at_index"]
    elif "age_at_diagnosis" in df.columns:
        df["age"] = df["age_at_diagnosis"]
    else:
        print("Warning: neither 'age_at_index' nor 'age_at_diagnosis' found -- "
              "no 'age' column will be available as a Cox covariate.")

    required = {"vital_status", "days_to_death", "days_to_last_follow_up"}
    missing = required - set(df.columns)
    if missing:
        print(f"Warning: clinical.csv is missing {missing} -- "
              "derive_survival_time_event() will fail without these. Check "
              "GDCquery_clinic()'s actual column names for this GDC data release.")

    if "stage_raw" in df.columns:
        # ajcc_pathologic_stage is a string like 'Stage IIIB' -- extract the
        # roman-numeral stage (I/II/III/IV), ignoring the trailing substage
        # letter, and collapse to a numeric ordinal for use as a Cox covariate.
        import re
        stage_map = {"IV": 4, "III": 3, "II": 2, "I": 1}  # order matters: longest match first

        def _to_numeric_stage(s):
            if not isinstance(s, str):
                return np.nan
            m = re.search(r"(IV|III|II|I)", s.upper())
            return stage_map[m.group(1)] if m else np.nan

        df["stage_numeric"] = df["stage_raw"].apply(_to_numeric_stage)

    if "patient_id" not in df.columns:
        id_col = next((c for c in ("bcr_patient_barcode", "submitter_id", "case_id") if c in df.columns), None)
        if id_col:
            df["patient_id"] = df[id_col].astype(str).str[:12]
        else:
            print("Warning: no obvious patient-id column found; you'll need to set "
                  "df['patient_id'] manually before joining to expression data.")

    return df.set_index("patient_id") if "patient_id" in df.columns else df


def load_r_exported_mutations(csv_path: str, gene_symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load mutations_maf.csv as written by assemble_tcga_brca_data.R. Thin
    wrapper that just applies the same gene filter as load_combined_maf()."""
    df = pd.read_csv(csv_path, low_memory=False)
    if gene_symbols and "Hugo_Symbol" in df.columns:
        df = df[df["Hugo_Symbol"].isin(gene_symbols)]
    return df


if __name__ == "__main__":
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "./fake_gdc_test"
    _run_smoke_tests(test_dir)
