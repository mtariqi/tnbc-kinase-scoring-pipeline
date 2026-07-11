# Data Provenance

Raw and intermediate data files are **not** committed to this repository. This is intentional:

- **TCGA-BRCA** expression data is ~21GB uncompressed — too large for git, and better fetched fresh from the authoritative source than kept as a stale copy.
- **DepMap** files require accepting DepMap's data use agreement at download time; redistributing them directly would bypass that.
- Everything below is fully reproducible from public sources using the scripts in `src/`.

## How to regenerate each dataset

### TCGA-BRCA (expression, clinical, mutations)
1. Download via TCGAbiolinks (R) — see the `GDCquery()`/`GDCdownload()` pattern in `src/gdc_local_data_loader.py`'s module docstring.
2. Build the expression matrix with `src/gdc_local_data_loader.py::build_expression_matrix()` (memory-safe, file-by-file — does not require `GDCprepare()`).
3. Fetch clinical data directly via `GDCquery_clinic()` (R).

### DepMap essentiality
1. Download `CRISPRGeneEffect.csv` and `Model.csv` from [depmap.org/portal/download](https://depmap.org/portal/download) (requires free registration + accepting the DUA).
2. Filter to your cell-line population of interest using `Model.csv`'s `OncotreeLineage`/`ModelSubtypeFeatures` columns.
3. Parse with `src/kinase_data_fetchers.py::parse_depmap_gene_effect()`.

### STRING network
Live API, no download needed — `src/string_network_builder.py::build_string_edges_communities_and_centrality()`.

### DGIdb drug-target interactions
Live API, no download needed — `src/kinase_data_fetchers.py::fetch_dgidb_interactions()`.

### openFDA FAERS toxicity data
Live API, no download needed — `src/kinase_data_fetchers.py::fetch_faers_reaction_profile()`.

## Directory layout once regenerated locally

```
data/
├── raw/
│   ├── kinases/kinase_90_list.txt
│   ├── drugs/drug_list.txt
│   ├── depmap/{CRISPRGeneEffect.csv, Model.csv, tnbc_model_ids.txt}
│   └── tcga_brca/GDCdata/...
└── processed/
    ├── tcga_brca/{expression_tpm.csv, clinical.csv, survival_stats.tsv}
    ├── depmap/depmap_tnbc_essentiality.tsv
    ├── string/{string_edges.tsv, community_map.tsv, centrality.tsv}
    ├── dgidb/dgidb_interactions.tsv
    └── faers/*.tsv
```

Run `python3 src/validate_pipeline_inputs.py` after regenerating to confirm everything is correctly shaped before computing CTS.
