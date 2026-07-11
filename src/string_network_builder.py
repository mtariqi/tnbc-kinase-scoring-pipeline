"""
STRING Network + Community Detection
=====================================

Produces the two files validate_pipeline_inputs.py currently reports as
missing:
    - string_edges.tsv    (source, target, weight) -- via STRING's live API
    - community_map.tsv   (kinase_id, community)    -- via Louvain community
                                                        detection on that network

STRING API note (verified against string-db.org/help/api/ directly, not
from memory): the LIVE API's 'score' field is already a float in [0,1].
The separately-downloadable BULK file (protein.links.txt.gz) uses an
integer x1000 scale instead -- these are two different things. This module
uses the live API, so no /1000 conversion is applied here. If you ever
load a bulk-downloaded protein.links file instead, you MUST divide by
1000 first (this is exactly the unnormalized-weight bug
validate_pipeline_inputs.py's validate_string_edges() checks for).

DEPENDENCY: networkx (already a dependency of kinase_scoring_pipeline.py's
ecosystem). Louvain community detection is built into networkx >= 2.8 as
nx.algorithms.community.louvain_communities -- no separate python-louvain
package needed.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import requests

STRING_API_BASE = "https://version-12-0.string-db.org/api"


# =====================================================================
# 1. RESOLVE GENE SYMBOLS -> STRING IDENTIFIERS
# =====================================================================

def string_map_ids(genes: List[str], species: int = 9606) -> Dict[str, str]:
    """
    Resolve gene symbols to their canonical STRING identifiers
    (e.g. 'EGFR' -> '9606.ENSP00000275493'). Always do this before calling
    fetch_string_network() -- querying by raw gene symbol can silently
    match the wrong protein for ambiguous names.

    Returns {input_gene_symbol: string_id}. Genes with no confident match
    are omitted (a warning is printed).
    """
    url = f"{STRING_API_BASE}/tsv/get_string_ids"
    params = {
        "identifiers": "\r".join(genes),
        "species": species,
        "limit": 1,  # best match only
        "caller_identity": "rtk_nrtk_tnbc_pipeline",
    }
    resp = requests.post(url, data=params, timeout=30)
    resp.raise_for_status()

    lines = [l for l in resp.text.strip().splitlines() if l]
    if not lines:
        return {}
    header = lines[0].split("\t")
    rows = [dict(zip(header, l.split("\t"))) for l in lines[1:]]

    mapping = {}
    for row in rows:
        query = row.get("queryItem") or row.get("preferredName")
        string_id = row.get("stringId")
        if query and string_id:
            mapping[query] = string_id

    missing = set(genes) - set(mapping.keys())
    if missing:
        print(f"Warning: {len(missing)} genes had no confident STRING ID match: {sorted(missing)}")
    return mapping


# =====================================================================
# 2. FETCH THE NETWORK AMONG A SET OF KINASES
# =====================================================================

def fetch_string_network(
    genes: List[str],
    species: int = 9606,
    required_score: int = 400,
    batch_size: int = 50,
    sleep_between_batches: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch all pairwise STRING interactions among `genes` (e.g. your 90
    RTK/NRTK kinases) at or above `required_score` (0-1000 scale for this
    PARAMETER specifically -- STRING's threshold parameter uses the x1000
    integer convention regardless of the response format; 400=medium
    confidence, 700=high, 900=highest -- this is STRING's own convention,
    not the same as the response 'score' column, which IS on [0,1]).

    Returns a DataFrame with columns: source, target, weight (weight =
    combined score, already on [0,1] from the live API -- see module
    docstring). Only genes actually present in `genes` appear as source/
    target (STRING's default network method restricts to the queried set
    when >1 identifier is given, so no filtering step is needed here).
    """
    id_map = string_map_ids(genes, species=species)
    if not id_map:
        raise ValueError("No genes resolved to STRING IDs -- check gene symbols/species.")

    string_ids = list(id_map.values())
    id_to_gene = {v: k for k, v in id_map.items()}

    all_rows = []
    for i in range(0, len(string_ids), batch_size):
        batch = string_ids[i : i + batch_size]
        url = f"{STRING_API_BASE}/tsv/network"
        params = {
            "identifiers": "\r".join(batch),
            "species": species,
            "required_score": required_score,
            "caller_identity": "rtk_nrtk_tnbc_pipeline",
        }
        resp = requests.post(url, data=params, timeout=60)
        resp.raise_for_status()

        lines = [l for l in resp.text.strip().splitlines() if l]
        if not lines:
            continue
        header = lines[0].split("\t")
        for line in lines[1:]:
            row = dict(zip(header, line.split("\t")))
            source_id = row.get("stringId_A")
            target_id = row.get("stringId_B")
            score = row.get("score")
            if source_id in id_to_gene and target_id in id_to_gene and score is not None:
                all_rows.append(
                    {
                        "source": id_to_gene[source_id],
                        "target": id_to_gene[target_id],
                        "weight": float(score),
                    }
                )

        if i + batch_size < len(string_ids):
            time.sleep(sleep_between_batches)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    # STRING returns each edge twice (A->B and B->A); collapse to one row per
    # unordered pair, keeping the max score if they ever differ slightly.
    df["pair_key"] = df.apply(lambda r: tuple(sorted([r["source"], r["target"]])), axis=1)
    df = df.loc[df.groupby("pair_key")["weight"].idxmax()].drop(columns="pair_key")
    return df.reset_index(drop=True)


# =====================================================================
# 3. LOUVAIN COMMUNITY DETECTION
# =====================================================================

def detect_communities(
    edges_df: pd.DataFrame,
    all_kinases: Optional[List[str]] = None,
    resolution: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run Louvain community detection on the crosstalk network to produce
    community_map.tsv's content.

    edges_df: columns source, target, weight (as from fetch_string_network())
    all_kinases: if given, any kinase with no edges at all is still included,
                 assigned its own singleton community (so community_map.tsv
                 has full coverage even for isolated nodes -- otherwise
                 validate_community_map()'s coverage check will fail on them).

    Returns a DataFrame with columns: kinase_id, community (int).
    """
    G = nx.Graph()
    if all_kinases:
        G.add_nodes_from(all_kinases)
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])

    communities = nx.algorithms.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)

    rows = []
    for community_id, members in enumerate(communities):
        for kinase in members:
            rows.append({"kinase_id": kinase, "community": community_id})

    return pd.DataFrame(rows).sort_values("kinase_id").reset_index(drop=True)


# =====================================================================
# 3b. CENTRALITY METRICS (betweenness / pagerank / degree) -- for CTS
# =====================================================================

def compute_centrality(
    edges_df: pd.DataFrame,
    all_kinases: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute the three network centrality metrics compute_cts() expects
    (betweenness, pagerank, degree) from the STRING crosstalk network.

    edges_df: columns source, target, weight (as from fetch_string_network())
    all_kinases: if given, kinases with no edges at all still appear in the
                 output with betweenness=pagerank=degree=0 (which
                 compute_cts()'s missing-value policy correctly treats as
                 "missing -> neutral", per the confirmed resolution table --
                 NOT silently dropped from the CTS computation).

    Returns a DataFrame with columns: kinase_id, betweenness, pagerank, degree.
    Ready to write directly to centrality.tsv.
    """
    G = nx.Graph()
    if all_kinases:
        G.add_nodes_from(all_kinases)
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])

    betweenness = nx.betweenness_centrality(G, weight="weight")
    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}
    degree = dict(G.degree(weight="weight"))

    df = pd.DataFrame({
        "kinase_id": list(G.nodes()),
    })
    df["betweenness"] = df["kinase_id"].map(betweenness)
    df["pagerank"] = df["kinase_id"].map(pagerank)
    df["degree"] = df["kinase_id"].map(degree)
    return df.sort_values("kinase_id").reset_index(drop=True)

def build_string_edges_and_communities(
    genes: List[str],
    species: int = 9606,
    required_score: int = 400,
    resolution: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the network and detect communities in one call. Returns
    (edges_df, community_df) ready to write straight to string_edges.tsv
    and community_map.tsv."""
    edges_df = fetch_string_network(genes, species=species, required_score=required_score)
    community_df = detect_communities(edges_df, all_kinases=genes, resolution=resolution)
    return edges_df, community_df


def build_string_edges_communities_and_centrality(
    genes: List[str],
    species: int = 9606,
    required_score: int = 400,
    resolution: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same as build_string_edges_and_communities(), plus centrality.tsv's
    content in one call. Returns (edges_df, community_df, centrality_df) --
    this is the one to use if you also need compute_cts()'s centrality input,
    which is the usual case."""
    edges_df = fetch_string_network(genes, species=species, required_score=required_score)
    community_df = detect_communities(edges_df, all_kinases=genes, resolution=resolution)
    centrality_df = compute_centrality(edges_df, all_kinases=genes)
    return edges_df, community_df, centrality_df


# =====================================================================
# 5. OFFLINE SMOKE TESTS -- MOCKED STRING RESPONSES + REAL COMMUNITY DETECTION
# =====================================================================

def _run_offline_smoke_tests() -> None:
    from unittest.mock import patch, MagicMock

    print("=== Testing string_map_ids() with a mocked response ===")
    fake_map_response = (
        "queryIndex\tstringId\tpreferredName\tqueryItem\r\n"
        "0\t9606.ENSP00000275493\tEGFR\tEGFR\r\n"
        "1\t9606.ENSP00000269571\tERBB2\tERBB2\r\n"
    )
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.text = fake_map_response
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        id_map = string_map_ids(["EGFR", "ERBB2"])
        print(f"  {id_map}\n")
        assert id_map == {"EGFR": "9606.ENSP00000275493", "ERBB2": "9606.ENSP00000269571"}

    print("=== Testing fetch_string_network() with a mocked response (incl. reverse-duplicate edge) ===")
    fake_map_response_3 = (
        "queryIndex\tstringId\tpreferredName\tqueryItem\r\n"
        "0\t9606.ENSP00000275493\tEGFR\tEGFR\r\n"
        "1\t9606.ENSP00000269571\tERBB2\tERBB2\r\n"
        "2\t9606.ENSP00000351276\tSRC\tSRC\r\n"
    )
    fake_network_response = (
        "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tscore\r\n"
        "9606.ENSP00000275493\t9606.ENSP00000269571\tEGFR\tERBB2\t0.92\r\n"
        "9606.ENSP00000269571\t9606.ENSP00000275493\tERBB2\tEGFR\t0.92\r\n"  # reverse duplicate
        "9606.ENSP00000275493\t9606.ENSP00000351276\tEGFR\tSRC\t0.55\r\n"
    )
    with patch("requests.post") as mock_post:
        def side_effect(url, data=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.text = fake_map_response_3 if "get_string_ids" in url else fake_network_response
            return resp
        mock_post.side_effect = side_effect
        edges_df = fetch_string_network(["EGFR", "ERBB2", "SRC"])
        print(edges_df, "\n")
        assert len(edges_df) == 2, f"expected 2 unique edges after dedup, got {len(edges_df)}"
        assert edges_df["weight"].between(0, 1).all(), "weights should already be on [0,1] from the live API"

    print("=== Testing detect_communities() on a network with OBVIOUS 2-community structure ===")
    # Two tight triangles (EGFR/ERBB2/ERBB4 and JAK1/JAK2/TYK2) connected by
    # one weak bridge edge -- Louvain should cleanly find 2 communities.
    synthetic_edges = pd.DataFrame([
        {"source": "EGFR", "target": "ERBB2", "weight": 0.9},
        {"source": "ERBB2", "target": "ERBB4", "weight": 0.9},
        {"source": "EGFR", "target": "ERBB4", "weight": 0.9},
        {"source": "JAK1", "target": "JAK2", "weight": 0.9},
        {"source": "JAK2", "target": "TYK2", "weight": 0.9},
        {"source": "JAK1", "target": "TYK2", "weight": 0.9},
        {"source": "ERBB2", "target": "JAK1", "weight": 0.05},  # weak bridge
    ])
    all_kinases = ["EGFR", "ERBB2", "ERBB4", "JAK1", "JAK2", "TYK2", "ISOLATED_KINASE"]
    community_df = detect_communities(synthetic_edges, all_kinases=all_kinases)
    print(community_df, "\n")

    assert community_df["kinase_id"].nunique() == len(all_kinases), "every kinase should appear, including isolated ones"
    egfr_group = community_df.loc[community_df["kinase_id"] == "EGFR", "community"].iloc[0]
    erbb4_group = community_df.loc[community_df["kinase_id"] == "ERBB4", "community"].iloc[0]
    jak1_group = community_df.loc[community_df["kinase_id"] == "JAK1", "community"].iloc[0]
    tyk2_group = community_df.loc[community_df["kinase_id"] == "TYK2", "community"].iloc[0]
    assert egfr_group == erbb4_group, "EGFR and ERBB4 (tight triangle) should be in the same community"
    assert jak1_group == tyk2_group, "JAK1 and TYK2 (tight triangle) should be in the same community"
    assert egfr_group != jak1_group, "the two triangles (only weakly bridged) should be DIFFERENT communities"
    assert community_df["community"].nunique() >= 2, "isolated kinase should get its own community, at minimum 2-3 total"

    print("  PASSED: the two tight triangles were correctly split into different")
    print("  communities, isolated node got its own community, weak bridge edge")
    print("  did not falsely merge them.\n")

    print("All offline smoke tests passed.")


if __name__ == "__main__":
    _run_offline_smoke_tests()
