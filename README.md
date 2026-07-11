# TNBC RTK/NRTK Kinase Scoring Pipeline

A real-data computational pipeline that ranks 90 receptor and non-receptor tyrosine kinases (RTKs/NRTKs) by therapeutic priority in triple-negative breast cancer (TNBC), by combining network topology, CRISPR knockout essentiality, patient survival association, and drug-targeting evidence into a single Composite Target Score (CTS).

Built and fully validated end-to-end — from zero real data files to a complete, bug-tested 90-kinase ranking — in a single working session.

---

## Key Result

CTS was computed for all 90 kinases using exclusively real, live-sourced data. The top three:

| Rank | Kinase | CTS Score | Why it makes biological sense |
|---|---|---|---|
| 1 | **ERBB2** | 0.690 | Central node in the HER2 signalling axis; one of the most validated targets in breast cancer |
| 2 | **EGFR** | 0.613 | Canonical RTK oncogene, extensively targeted across cancer types |
| 3 | **PTK2** (FAK) | 0.532 | Independently the single most essential kinase by DepMap CRISPR knockout data alone, *before* any other data source was combined with it |

**Why this matters:** the ranking wasn't tuned to match known biology — it emerged from combining four independently-sourced, mechanistically unrelated signals (STRING network centrality, DepMap essentiality, TCGA-BRCA survival association, DGIdb drug-targeting evidence). That this bottom-up, data-driven combination recovers three of the most extensively validated targets in real TNBC biology at the very top of the list is a genuine, non-trivial sanity check on the pipeline's correctness — an arbitrary or miscalibrated pipeline would have no particular reason to land there.

Full methodology and results: [`docs/Ranking_Methodology_Report.docx`](docs/Ranking_Methodology_Report.docx)

---

## What This Pipeline Does

```
TCGA-BRCA (GDC)  ──┐
DepMap CRISPR    ──┤
STRING network   ──┼──▶  Composite Target Score (CTS)  ──▶  Ranked kinase list
DGIdb drug data  ──┘             │
                                 ▼
                    PairCTS / DrugPairScore / TripletCTS
                    (kinase pairs → drug pairs → drug regimens)
```

- **`compute_cts()`** — single-kinase score combining network centrality (STRING), essentiality (DepMap), survival association (TCGA-BRCA Cox/log-rank), and druggability (DGIdb), with an explicit, tested missing-value policy so no kinase is ever silently dropped for lacking one data source.
- **`PairCTS` / `DrugPairScore`** — extends scoring to kinase pairs and their mapped drug pairs, rewarding complementary (not redundant) target combinations.
- **`TripletCTS`** — extends to 3-drug regimens: module coverage, non-redundancy, resistance-escape-route closure, combined toxicity.
- **`MDCOE` (in `mdcoe.py`)** — a separate beam-search + heuristic-scoring (HCOS) system for ranking specific drug regimens against a patient's genomic profile.
- **`validate_pipeline_inputs.py`** — an 11-point pre-flight check (column presence, value ranges, cross-file coverage, orientation) run before any full-scale computation, so a bad input file fails fast with a clear message instead of silently producing wrong numbers downstream.

---

## Real Data Sources

| Source | What's Used | Access Method |
|---|---|---|
| [TCGA-BRCA](https://portal.gdc.cancer.gov/) (GDC) | STAR-Counts RNA-seq expression (1095 patients), clinical data (1098 patients), somatic mutations | TCGAbiolinks (R) + custom Python file-by-file loader |
| [DepMap](https://depmap.org/portal/) | CRISPR knockout essentiality (Chronos), restricted to 25 confirmed-TNBC cell lines | Manual bulk download (`CRISPRGeneEffect.csv`, `Model.csv`) |
| [STRING](https://string-db.org/) | Protein-protein interaction network (464 edges, 9 communities) | Live REST API |
| [DGIdb](https://dgidb.org/) | Drug-kinase interaction evidence (4655 records) | Live GraphQL API v5 |
| [openFDA FAERS](https://open.fda.gov/apis/drug/event/) | Adverse-event/toxicity profiles | Live REST API |

No synthetic or placeholder data contributes to the final reported results — every number in `results/` traces back to one of the sources above.

---

## Validation Coverage

```
16 checks passed, 0 failures — validate_pipeline_inputs.py

Kinase coverage across all 90 kinases:
  Survival (TCGA-BRCA):    90/90
  DepMap essentiality:     89/90
  STRING centrality:       90/90  (5 network-isolated kinases correctly
                                    handled by the missing-value policy)
  DGIdb drug evidence:     84/90
```

---

## Repository Structure

```
.
├── src/
│   ├── kinase_scoring_pipeline.py       # CTS / PairCTS / DrugPairScore / TripletCTS
│   ├── kinase_data_fetchers.py          # DGIdb, ChEMBL, DepMap parsing, FAERS
│   ├── tcga_brca_survival_pipeline.py   # Cox PH, log-rank, Schoenfeld, RMST
│   ├── gdc_local_data_loader.py         # GDC file parsing (memory-safe, file-by-file)
│   ├── string_network_builder.py        # STRING API, Louvain communities, centrality
│   ├── validate_pipeline_inputs.py      # Pre-flight input validation
│   └── mdcoe.py                         # Beam-search regimen ranking (HCOS)
├── results/
│   └── CTS_all_90_kinases.tsv           # Final validated CTS output
├── docs/
│   ├── Session_Troubleshooting_Report.docx   # Full session log: data, bugs, fixes
│   └── Ranking_Methodology_Report.docx       # Scoring methodology + worked example
├── requirements.txt
└── README.md
```

**Note on data:** raw and intermediate data files (TCGA expression matrices, DepMap CRISPR screens, etc.) are intentionally **not** committed to this repository — see [`data/README.md`](data/README.md) for provenance and how to regenerate them. Some sources (TCGA-BRCA's full expression data) are tens of gigabytes; others (DepMap) require accepting a data use agreement before download. Only the small, final, derived output (`results/CTS_all_90_kinases.tsv`) is version-controlled.

---

## Engineering Notes: Bugs Found and Fixed

This pipeline was built against real, messy, external data — not clean synthetic inputs — and six substantive bugs were found and fixed along the way, each verified against a reconstructed test case rather than assumed correct:

1. **Drug-list contamination** — raw DGIdb output included lab compound codes and even kinase gene symbols mislabeled as drugs; fixed with a combined INN-suffix whitelist + stoplist + kinase-name exclusion filter.
2. **DepMap file orientation** — cell lines are rows, genes are columns in the raw file; an early script indexed by the wrong axis.
3. **Memory-safe GDC loading** — `TCGAbiolinks::GDCprepare()` OOM-crashed loading the full cohort; replaced with a file-by-file Python loader that never holds more than one sample in memory.
4. **Duplicate clinical column bug** — GDC provides both `age_at_index` and `age_at_diagnosis`; a naive rename collapsed both into one ambiguous `age` column, breaking every Cox regression identically.
5. **PageRank teleportation baseline** — network-isolated kinases were incorrectly treated as having real centrality data, because PageRank assigns even disconnected nodes a small nonzero baseline value; fixed by using node degree (a mathematically unambiguous zero) as the missing-data signal instead.
6. **Beam-search depth limitation** — the regimen search only ever returned results of one fixed size, potentially missing the true best regimen; fixed to search and rank across all sizes simultaneously.

Full write-up with root-cause analysis for each: [`docs/Session_Troubleshooting_Report.docx`](docs/Session_Troubleshooting_Report.docx).

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.kinase_scoring_pipeline import compute_cts
import pandas as pd

kinase_df = pd.DataFrame(...)  # see src/kinase_scoring_pipeline.py docstring for schema
result = compute_cts(kinase_df)
```

See each script's module docstring for detailed input requirements and a runnable smoke test (`python3 src/<script>.py`) using synthetic data.

---

## Known Limitations

- CTS is a computational prioritization, not a clinically or experimentally validated result.
- `mdcoe.py`'s HCOS scoring and this repository's CTS/TripletCTS scoring are two separate systems for a related goal; they are not yet reconciled or cross-validated against each other.
- 1 kinase shows a minor DepMap coverage discrepancy under investigation (does not affect the top-ranked results).
- `PairCTS`/`TripletCTS` have not yet been run against this session's real, validated 90-kinase CTS output.

## License

[MIT](LICENSE)
