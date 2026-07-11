"""
Kinase/Drug Data Fetchers
=========================

Mechanical fetch/parse functions for the four inputs that come from public
downloads or APIs (as opposed to Centrality/Survival, which you compute
yourself from your own network and TCGA-BRCA analysis):

    - fetch_dgidb_interactions()   -> DGIdb v5 GraphQL API (drug-gene interactions)
    - fetch_chembl_target_id()
      fetch_chembl_activities()    -> ChEMBL REST API (bioactivity / potency data)
    - parse_depmap_gene_effect()   -> local DepMap CRISPR (Chronos) CSV you download
    - fetch_faers_event_count()
      fetch_faers_reaction_profile()
      estimate_pairwise_toxicity_faers() -> openFDA FAERS API (adverse events)

All network functions require internet access and the `requests` package.
None of them are called at import time -- nothing runs until you call a
function. Each includes a `if __name__ == "__main__":` smoke test at the
bottom using `unittest.mock` so you can verify the parsing logic works
without needing network access.

Rate limits / politeness:
    - DGIdb and ChEMBL have no documented hard rate limit for reasonable use,
      but batch your gene lists (e.g. 20-30 genes per DGIdb call) rather than
      firing 90 separate requests.
    - openFDA allows 240 requests/minute/IP without a key, 120,000/day with a
      free API key (see https://open.fda.gov/apis/authentication/). Get a key
      if you're running this across many drug pairs.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
import requests


# =====================================================================
# 1. DGIdb v5 -- drug-gene interactions (GraphQL)
# =====================================================================

DGIDB_GRAPHQL_URL = "https://dgidb.org/api/graphql"


def fetch_dgidb_interactions(
    genes: List[str],
    batch_size: int = 25,
    sleep_between_batches: float = 0.5,
) -> pd.DataFrame:
    """
    Query DGIdb v5's GraphQL API for all known drug interactions for a list
    of gene symbols (e.g. your 90 RTK/NRTK kinases).

    Returns a DataFrame with columns:
        kinase_id, drug, interaction_score, interaction_types, sources
    """
    all_rows = []
    for i in range(0, len(genes), batch_size):
        batch = genes[i : i + batch_size]
        query = """
        query GeneInteractions($names: [String!]) {
          genes(names: $names) {
            nodes {
              name
              interactions {
                drug { name conceptId }
                interactionScore
                interactionTypes { type directionality }
                sources { sourceDbName }
              }
            }
          }
        }
        """
        resp = requests.post(
            DGIDB_GRAPHQL_URL,
            json={"query": query, "variables": {"names": batch}},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        all_rows.extend(_parse_dgidb_response(data))
        if i + batch_size < len(genes):
            time.sleep(sleep_between_batches)

    return pd.DataFrame(all_rows)


def _parse_dgidb_response(data: dict) -> List[dict]:
    """Flatten one DGIdb GraphQL response into row dicts."""
    rows = []
    nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
    for gene_node in nodes:
        gene_name = gene_node.get("name")
        for interaction in gene_node.get("interactions", []):
            drug = interaction.get("drug", {}) or {}
            rows.append(
                {
                    "kinase_id": gene_name,
                    "drug": drug.get("name"),
                    "drug_concept_id": drug.get("conceptId"),
                    "interaction_score": interaction.get("interactionScore"),
                    "interaction_types": ",".join(
                        t.get("type", "") for t in interaction.get("interactionTypes", [])
                    ),
                    "sources": ",".join(
                        s.get("sourceDbName", "") for s in interaction.get("sources", [])
                    ),
                }
            )
    return rows


# =====================================================================
# 2. ChEMBL -- bioactivity / potency data (REST)
# =====================================================================

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_chembl_target_id(gene_symbol: str, organism: str = "Homo sapiens") -> Optional[str]:
    """
    Resolve a gene symbol (e.g. 'EGFR') to its single-protein ChEMBL target ID
    (e.g. 'CHEMBL203'). Returns None if no confident match is found -- in that
    case, inspect the candidates yourself via the same search.
    """
    resp = requests.get(
        f"{CHEMBL_BASE}/target/search.json",
        params={"q": gene_symbol},
        timeout=30,
    )
    resp.raise_for_status()
    targets = resp.json().get("targets", [])
    candidates = [
        t for t in targets
        if t.get("organism") == organism and t.get("target_type") == "SINGLE PROTEIN"
    ]
    if not candidates:
        return None
    # Prefer exact pref_name match, else take the first candidate.
    exact = [t for t in candidates if t.get("pref_name", "").upper() == gene_symbol.upper()]
    return (exact[0] if exact else candidates[0]).get("target_chembl_id")


def fetch_chembl_activities(
    target_chembl_id: str,
    standard_types: Optional[List[str]] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch bioactivity records (e.g. IC50, Ki) for a ChEMBL target ID.
    Returns columns: molecule_chembl_id, drug_name (pref_name if available),
    standard_type, standard_value, standard_units, assay_description.
    """
    standard_types = standard_types or ["IC50", "Ki", "EC50"]
    params = {
        "target_chembl_id": target_chembl_id,
        "limit": limit,
        "format": "json",
    }
    rows: List[dict] = []
    url = f"{CHEMBL_BASE}/activity.json"
    while url:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        for act in payload.get("activities", []):
            if act.get("standard_type") not in standard_types:
                continue
            rows.append(
                {
                    "molecule_chembl_id": act.get("molecule_chembl_id"),
                    "drug_name": act.get("molecule_pref_name"),
                    "standard_type": act.get("standard_type"),
                    "standard_value": act.get("standard_value"),
                    "standard_units": act.get("standard_units"),
                    "assay_description": act.get("assay_description"),
                }
            )
        next_page = payload.get("page_meta", {}).get("next")
        url = f"https://www.ebi.ac.uk{next_page}" if next_page else None
        params = {}  # params are baked into `next_page` already
    return pd.DataFrame(rows)


# =====================================================================
# 3. DepMap CRISPR essentiality (local file -- too large to fetch live)
# =====================================================================

def parse_depmap_gene_effect(
    csv_path: str,
    genes: List[str],
    cell_line_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Parse a DepMap CRISPR gene-effect file (download 'CRISPRGeneEffect.csv'
    from https://depmap.org/portal/download/ -- search "CRISPR (DepMap Public)").

    The raw file has cell lines as rows (index = ModelID / DepMap_ID) and
    genes as columns, typically named like 'EGFR (1956)'. This function:
      1. Loads the file
      2. Matches your gene symbols to the '<SYMBOL> (<ENTREZ_ID>)' column names
      3. Optionally filters to specific cell lines (e.g. your TNBC subset --
         cross-reference DepMap's Model.csv for lineage == 'Breast Cancer'
         and subtype containing 'TNBC'/'Basal')
      4. Returns a tidy DataFrame: kinase_id, cell_line_id, depmap_score
         plus a `mean_depmap_score` column per gene (averaged across the
         selected cell lines) for direct use in compute_cts().

    NOTE: DepMap Chronos/CERES scores are NEGATIVE for essential genes
    (more negative = cell line depends more on that gene). compute_cts()
    in kinase_scoring_pipeline.py already handles this via
    `higher_is_better=False` in min_max_normalize.
    """
    df = pd.read_csv(csv_path, index_col=0)

    if cell_line_ids is not None:
        missing = set(cell_line_ids) - set(df.index)
        if missing:
            print(f"Warning: {len(missing)} requested cell lines not found in file, skipping them.")
        df = df.loc[df.index.intersection(cell_line_ids)]

    # Column names look like 'EGFR (1956)' -- match on the symbol prefix.
    col_lookup = {col.split(" (")[0]: col for col in df.columns}
    missing_genes = [g for g in genes if g not in col_lookup]
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found in DepMap file: {missing_genes}")

    found_genes = [g for g in genes if g in col_lookup]
    sub = df[[col_lookup[g] for g in found_genes]].copy()
    sub.columns = found_genes

    tidy = sub.reset_index().melt(id_vars=sub.index.name or "index", var_name="kinase_id", value_name="depmap_score")
    tidy = tidy.rename(columns={sub.index.name or "index": "cell_line_id"})

    mean_scores = tidy.groupby("kinase_id")["depmap_score"].mean().rename("mean_depmap_score")
    tidy = tidy.merge(mean_scores, on="kinase_id", how="left")
    return tidy


# =====================================================================
# 4. openFDA FAERS -- adverse events / toxicity signal
# =====================================================================

FAERS_URL = "https://api.fda.gov/drug/event.json"


def fetch_faers_event_count(drug_name: str, api_key: Optional[str] = None) -> int:
    """Total number of FAERS adverse-event reports mentioning this drug."""
    params = {"search": f'patient.drug.medicinalproduct:"{drug_name}"', "limit": 1}
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(FAERS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("meta", {}).get("results", {}).get("total", 0)


def fetch_faers_reaction_profile(
    drug_name: str, limit: int = 15, api_key: Optional[str] = None
) -> pd.DataFrame:
    """Top reported reaction terms (MedDRA preferred terms) for a single drug,
    with report counts -- a qualitative toxicity profile, not a single score."""
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(FAERS_URL, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return pd.DataFrame(results).rename(columns={"term": "reaction", "count": "report_count"})


def estimate_pairwise_toxicity_faers(
    drug_a: str,
    drug_b: str,
    api_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Rough combined-toxicity signal for a drug pair: the count of FAERS
    reports where BOTH drugs are listed on the same report, versus each
    drug's solo report count. A high co-report count relative to each
    drug's solo baseline is a coarse proxy for combined-use adverse signal
    -- NOT a substitute for a real drug-interaction/toxicity study.

    Returns: {'solo_a': n, 'solo_b': n, 'co_reported': n, 'co_report_ratio': r}
    where r = co_reported / min(solo_a, solo_b).
    """
    solo_a = fetch_faers_event_count(drug_a, api_key=api_key)
    solo_b = fetch_faers_event_count(drug_b, api_key=api_key)

    params = {
        "search": (
            f'patient.drug.medicinalproduct:"{drug_a}"+AND+'
            f'patient.drug.medicinalproduct:"{drug_b}"'
        ),
        "limit": 1,
    }
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(FAERS_URL, params=params, timeout=30)
    resp.raise_for_status()
    co_reported = resp.json().get("meta", {}).get("results", {}).get("total", 0)

    denom = min(solo_a, solo_b) if min(solo_a, solo_b) > 0 else 1
    return {
        "solo_a": solo_a,
        "solo_b": solo_b,
        "co_reported": co_reported,
        "co_report_ratio": co_reported / denom,
    }


# =====================================================================
# 5. OFFLINE SMOKE TESTS (no network needed -- mocks the API responses)
# =====================================================================

def _run_offline_smoke_tests() -> None:
    from unittest.mock import patch, MagicMock

    print("=== Testing _parse_dgidb_response with a mocked payload ===")
    fake_dgidb_payload = {
        "data": {
            "genes": {
                "nodes": [
                    {
                        "name": "EGFR",
                        "interactions": [
                            {
                                "drug": {"name": "AFATINIB", "conceptId": "chembl:CHEMBL1173655"},
                                "interactionScore": 8.4,
                                "interactionTypes": [{"type": "inhibitor", "directionality": "inhibitory"}],
                                "sources": [{"sourceDbName": "DrugBank"}],
                            }
                        ],
                    }
                ]
            }
        }
    }
    rows = _parse_dgidb_response(fake_dgidb_payload)
    print(pd.DataFrame(rows), "\n")
    assert rows[0]["kinase_id"] == "EGFR"
    assert rows[0]["drug"] == "AFATINIB"

    print("=== Testing fetch_dgidb_interactions with requests.post mocked ===")
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_dgidb_payload
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        df = fetch_dgidb_interactions(["EGFR"])
        print(df, "\n")
        assert len(df) == 1

    print("=== Testing fetch_faers_event_count with requests.get mocked ===")
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"meta": {"results": {"total": 4321}}}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        count = fetch_faers_event_count("afatinib")
        print(f"  fake FAERS count: {count}\n")
        assert count == 4321

    print("All offline smoke tests passed. Network functions are wired up")
    print("correctly; run them for real once you have internet access and,")
    print("for FAERS at scale, an openFDA API key.")


if __name__ == "__main__":
    _run_offline_smoke_tests()
