"""
mdcoe.py

Working implementation of the MDCOE beam search + HCOS scorer described in
Section 4.2.5 of the report. Built to run directly on top of the existing
tnbc_combo_pipeline.ipynb knowledge base (GENE_PATHWAYS, GENE_DRUGS,
DRUG_TOXICITY, KNOWN_SYNERGY_BOOST) so it plugs into real patient data rather
than a toy example.

HONESTY NOTE: like the rest of the pipeline, the underlying "SynergyNet" score
is still the documented placeholder heuristic, not a trained GNN. MDCOE here
is proving the SEARCH AND RANKING LOGIC works correctly over that heuristic,
not that the resulting regimens are clinically validated.
"""

import itertools

# ---- knowledge base (same as tnbc_combo_pipeline.ipynb) ----
GENE_PATHWAYS = {
    "EGFR": ["MAPK/ERK", "PI3K/AKT"], "ERBB2": ["MAPK/ERK", "PI3K/AKT"],
    "MET": ["MAPK/ERK", "PI3K/AKT"], "SRC": ["MAPK/ERK"], "ABL1": ["MAPK/ERK"],
    "FGFR1": ["MAPK/ERK", "PI3K/AKT"], "FGFR2": ["MAPK/ERK", "PI3K/AKT"],
    "FGFR3": ["MAPK/ERK", "PI3K/AKT"], "ALK": ["MAPK/ERK", "PI3K/AKT"],
    "KIT": ["MAPK/ERK", "PI3K/AKT"], "PDGFRA": ["MAPK/ERK", "PI3K/AKT"],
    "BRAF": ["MAPK/ERK"], "PIK3CA": ["PI3K/AKT"], "AKT1": ["PI3K/AKT"],
    "MTOR": ["PI3K/AKT"], "JAK2": ["JAK/STAT"], "BRCA1": ["DNA Damage Repair"],
    "BRCA2": ["DNA Damage Repair"], "CDK4": ["Cell Cycle"], "CDK6": ["Cell Cycle"],
    # TNBC-specific additions
    "TP53": ["Cell Cycle Checkpoint"],       # loss creates WEE1i synthetic-lethality
    "RB1": ["Cell Cycle"],                    # loss modifies CDK4/6i response; no direct inhibitor
    "MYC": ["MYC/Transcriptional"],           # classically "undruggable"; indirect (BET) targeting only
    "PTEN": ["PI3K/AKT"],                     # loss de-represses PI3K/AKT, same pathway as PIK3CA
    "AR": ["Androgen Signalling"],            # relevant to the LAR TNBC subtype
    # Full RTK (58) + NRTK (32) kinome additions, Robinson/Wu/Lin Oncogene 2000 classification
    "ERBB3": ["MAPK/ERK"], "ERBB4": ["MAPK/ERK"],
    "INSR": ["PI3K/AKT"], "IGF1R": ["PI3K/AKT"], "INSRR": ["PI3K/AKT"],
    "PDGFRB": ["MAPK/ERK"], "CSF1R": ["MAPK/ERK"],
    "FLT1": ["MAPK/ERK"], "KDR": ["MAPK/ERK"], "FLT4": ["MAPK/ERK"],
    "FGFR4": ["MAPK/ERK"],
    "PTK7": ["MAPK/ERK"],
    "NTRK1": ["MAPK/ERK"], "NTRK2": ["MAPK/ERK"], "NTRK3": ["MAPK/ERK"],
    "ROR1": ["MAPK/ERK"], "ROR2": ["MAPK/ERK"],
    "MUSK": ["MAPK/ERK"],
    "MST1R": ["MAPK/ERK"],
    "AXL": ["PI3K/AKT"], "MERTK": ["PI3K/AKT"], "TYRO3": ["PI3K/AKT"],
    "TEK": ["PI3K/AKT"], "TIE1": ["PI3K/AKT"],
    "EPHA1": ["MAPK/ERK"], "EPHA2": ["MAPK/ERK"], "EPHA3": ["MAPK/ERK"], "EPHA4": ["MAPK/ERK"],
    "EPHA5": ["MAPK/ERK"], "EPHA6": ["MAPK/ERK"], "EPHA7": ["MAPK/ERK"], "EPHA8": ["MAPK/ERK"],
    "EPHA10": ["MAPK/ERK"], "EPHB1": ["MAPK/ERK"], "EPHB2": ["MAPK/ERK"], "EPHB3": ["MAPK/ERK"],
    "EPHB4": ["MAPK/ERK"], "EPHB6": ["MAPK/ERK"],
    "RET": ["MAPK/ERK"],
    "ROS1": ["MAPK/ERK"],
    "LTK": ["MAPK/ERK"],
    "AATK": ["MAPK/ERK"], "LMTK2": ["MAPK/ERK"], "LMTK3": ["MAPK/ERK"],
    "DDR1": ["MAPK/ERK"], "DDR2": ["MAPK/ERK"],
    "RYK": ["MAPK/ERK"],
    "STYK1": ["MAPK/ERK"],
    "ABL2": ["MAPK/ERK"],
    "TNK2": ["MAPK/ERK"], "TNK1": ["MAPK/ERK"],
    "CSK": ["MAPK/ERK"], "MATK": ["MAPK/ERK"],
    "PTK2": ["PI3K/AKT"], "PTK2B": ["PI3K/AKT"],
    "FES": ["MAPK/ERK"], "FER": ["MAPK/ERK"],
    "FRK": ["MAPK/ERK"], "PTK6": ["MAPK/ERK"], "SRMS": ["MAPK/ERK"],
    "JAK1": ["JAK/STAT"], "JAK3": ["JAK/STAT"], "TYK2": ["JAK/STAT"],
    "FYN": ["MAPK/ERK"], "YES1": ["MAPK/ERK"], "FGR": ["MAPK/ERK"],
    "LCK": ["MAPK/ERK"], "HCK": ["MAPK/ERK"], "BLK": ["MAPK/ERK"], "LYN": ["MAPK/ERK"],
    "SYK": ["BCR/Immune Signalling"], "ZAP70": ["BCR/Immune Signalling"],
    "TEC": ["BCR/Immune Signalling"], "BTK": ["BCR/Immune Signalling"],
    "ITK": ["BCR/Immune Signalling"], "BMX": ["BCR/Immune Signalling"], "TXK": ["BCR/Immune Signalling"],
}

GENE_DRUGS = {
    "EGFR": ["erlotinib", "gefitinib", "afatinib"], "ERBB2": ["lapatinib", "neratinib", "trastuzumab"],
    "MET": ["crizotinib", "capmatinib"], "SRC": ["dasatinib", "bosutinib"],
    "ABL1": ["imatinib", "dasatinib", "nilotinib"], "FGFR1": ["erdafitinib", "pemigatinib"],
    "FGFR2": ["erdafitinib", "pemigatinib"], "FGFR3": ["erdafitinib", "pemigatinib"],
    "ALK": ["alectinib", "crizotinib"], "KIT": ["imatinib", "sunitinib"],
    "PDGFRA": ["imatinib", "sunitinib"], "BRAF": ["vemurafenib", "dabrafenib"],
    "PIK3CA": ["alpelisib"], "AKT1": ["capivasertib"], "MTOR": ["everolimus", "temsirolimus"],
    "JAK2": ["ruxolitinib"], "BRCA1": ["olaparib", "talazoparib"], "BRCA2": ["olaparib", "talazoparib"],
    "CDK4": ["palbociclib", "ribociclib"], "CDK6": ["palbociclib", "ribociclib"],
    # TNBC-specific additions
    "TP53": ["adavosertib"],          # WEE1i; exploits TP53-loss synthetic lethality, not a direct TP53 drug
    "RB1": [],                        # honestly no direct inhibitor; tracked for pathway annotation only
    "MYC": ["molibresib"],            # BET inhibitor; indirect MYC-pathway targeting, investigational
    "PTEN": ["capivasertib", "alpelisib"],  # loss activates PI3K/AKT, same agents as that pathway
    "AR": ["enzalutamide", "bicalutamide"],  # relevant to LAR TNBC subtype
    # RTK/NRTK kinome additions with real approved/investigational inhibitors only.
    # Most of the 90 new RTK/NRTK genes get NO entry here -- intentionally, since
    # most are orphan/undruggable (e.g. most EPH receptors, MUSK, PTK7, RYK, STYK1,
    # LMR family). Forcing a fake drug mapping would be worse than leaving it empty.
    "INSR": ["linsitinib"], "IGF1R": ["linsitinib"],
    "FLT3": ["midostaurin", "gilteritinib"],
    "FLT1": ["sunitinib", "sorafenib", "pazopanib", "axitinib"],
    "KDR": ["sunitinib", "sorafenib", "pazopanib", "axitinib", "lenvatinib"],
    "FLT4": ["axitinib", "pazopanib"],
    "CSF1R": ["pexidartinib"],
    "NTRK1": ["larotrectinib", "entrectinib"],
    "NTRK2": ["larotrectinib", "entrectinib"],
    "NTRK3": ["larotrectinib", "entrectinib"],
    "RET": ["selpercatinib", "pralsetinib"],
    "ROS1": ["crizotinib", "entrectinib"],
    "DDR1": ["nilotinib"], "DDR2": ["dasatinib"],
    "ABL2": ["imatinib", "nilotinib"],
    "JAK1": ["ruxolitinib", "baricitinib"],
    "JAK3": ["tofacitinib"],
    "TYK2": ["deucravacitinib"],
    "BTK": ["ibrutinib", "acalabrutinib", "zanubrutinib"],
    "SYK": ["fostamatinib"],
}

# TP53 zygosity-stratified strategy: a missense mutant protein and a fully deleted
# gene are mechanistically different problems and call for different drugs. A flat
# GENE_DRUGS["TP53"] lookup can't make this distinction -- this function can.
#   - wild_type: not applicable here (gene isn't "altered", so it never reaches this
#     function under the current alteration-driven architecture; MDM2-amplification-
#     triggered MDM2i strategy is a known, separate future extension, not yet built)
#   - mutant (missense, protein present but misfolded): WEE1i + mutant-p53 reactivator
#   - deleted (biallelic loss, no protein at all): WEE1i + BCL-2i (p53-independent)
TP53_ZYGOSITY_DRUGS = {
    "mutant": ["adavosertib", "eprenetapopt"],
    "deleted": ["adavosertib", "venetoclax"],
}


def resolve_tp53_drugs(alteration_type):
    """Map a TP53 alteration_type string (from MAF/CNV extraction) to the
    zygosity-appropriate drug list.

    Three-way, NMD-aware classification (refined from an earlier two-way version):
      - Stable mutant protein (missense, in-frame indel): the protein is present,
        just misfolded -- reactivators like eprenetapopt can bind it.
      - Truncating mutation (nonsense, frameshift, splice site): premature stop
        codons trigger nonsense-mediated decay (NMD), degrading the transcript
        before translation. TCGA pan-cancer analysis confirms truncating TP53
        mutations show REDUCED mRNA vs. wild-type/missense, consistent with NMD,
        while missense/in-frame mutations show stable or slightly increased mRNA.
        There is usually no stable protein for a reactivator to bind, so this
        case is treated the same as a deletion, not as a reactivatable mutant.
      - Deep/homozygous deletion (from CNV): no gene, no transcript, no protein.
    """
    deletion_types = {"Deep_Deletion", "Homozygous_Deletion"}
    truncating_types = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"}
    stable_mutant_types = {"Missense_Mutation", "In_Frame_Del", "In_Frame_Ins"}

    if alteration_type in deletion_types or alteration_type in truncating_types:
        # NMD-degraded truncating mutations functionally phenocopy deletion:
        # no stable protein exists for a reactivator to target either way.
        return TP53_ZYGOSITY_DRUGS["deleted"]
    if alteration_type in stable_mutant_types:
        return TP53_ZYGOSITY_DRUGS["mutant"]
    # unrecognised classification: default to the mutant bucket but this should
    # be reviewed manually rather than trusted silently
    return TP53_ZYGOSITY_DRUGS["mutant"]

DRUG_TOXICITY = {
    "erlotinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "afatinib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "crizotinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": False},
    "alpelisib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": True, "myelosuppressive": False},
    "olaparib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "talazoparib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "capivasertib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": True, "myelosuppressive": False},
    "dasatinib": {"hepatotoxic": False, "cardiotoxic": True, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": True},
    "everolimus": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": True, "myelosuppressive": True},
    "palbociclib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": True},
    "adavosertib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "molibresib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "enzalutamide": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "bicalutamide": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "eprenetapopt": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    # Filled after auditing GENE_DRUGS vs DRUG_TOXICITY and finding 36 drugs with
    # no toxicity entry at all -- meaning "no shared toxicity" for those pairs
    # meant "no data," not "confirmed safe." Real, label/literature-based values:
    "gefitinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "lapatinib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": False},
    "neratinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "trastuzumab": {"hepatotoxic": False, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "sunitinib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": True},
    "sorafenib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "pazopanib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": False},
    "axitinib": {"hepatotoxic": False, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "lenvatinib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": False},
    "imatinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "nilotinib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": True, "hyperglycemic": True, "myelosuppressive": True},
    "bosutinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "dabrafenib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "vemurafenib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": False},
    "alectinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "entrectinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": True, "myelosuppressive": False},
    "capmatinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "erdafitinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": True, "myelosuppressive": False},
    "pemigatinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": True, "myelosuppressive": True},
    "ribociclib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": True},
    "temsirolimus": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": True, "myelosuppressive": True},
    "ruxolitinib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "baricitinib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "tofacitinib": {"hepatotoxic": False, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "deucravacitinib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "linsitinib": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": True, "myelosuppressive": False},
    "midostaurin": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": True},
    "gilteritinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": True},
    "larotrectinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "selpercatinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": False},
    "pralsetinib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": True, "hyperglycemic": False, "myelosuppressive": True},
    "pexidartinib": {"hepatotoxic": True, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": False},
    "ibrutinib": {"hepatotoxic": False, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "acalabrutinib": {"hepatotoxic": False, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "zanubrutinib": {"hepatotoxic": False, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "fostamatinib": {"hepatotoxic": True, "cardiotoxic": True, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
    "venetoclax": {"hepatotoxic": False, "cardiotoxic": False, "qt_prolonging": False, "hyperglycemic": False, "myelosuppressive": True},
}

KNOWN_SYNERGY_BOOST = {
    frozenset(["erlotinib", "crizotinib"]): 0.25,
    frozenset(["olaparib", "alpelisib"]): 0.30,
    frozenset(["palbociclib", "alpelisib"]): 0.20,
    frozenset(["dasatinib", "everolimus"]): 0.15,
}

RTK = {"EGFR", "ERBB2", "MET", "FGFR1", "FGFR2", "FGFR3", "ALK", "KIT", "PDGFRA"}
INTRA = {"SRC", "ABL1", "BRAF", "PIK3CA", "AKT1", "MTOR", "JAK2", "CDK4", "CDK6"}

DRUG_TO_GENE = {}
for gene, drugs in GENE_DRUGS.items():
    for d in drugs:
        DRUG_TO_GENE.setdefault(d, gene)


# ---- SynergyNet: pairwise local score (same heuristic as the notebook) ----
def pairwise_synergy(drug_a, drug_b):
    gene_a, gene_b = DRUG_TO_GENE.get(drug_a), DRUG_TO_GENE.get(drug_b)
    if gene_a is None or gene_b is None or gene_a == gene_b:
        return 0.0
    overlap = len(set(GENE_PATHWAYS.get(gene_a, [])) & set(GENE_PATHWAYS.get(gene_b, [])))
    overlap_score = min(overlap * 0.15, 0.4)
    complementary = (gene_a in RTK and gene_b in INTRA) or (gene_b in RTK and gene_a in INTRA)
    complementarity_score = 0.25 if complementary else 0.1
    boost = KNOWN_SYNERGY_BOOST.get(frozenset([drug_a, drug_b]), 0.0)
    return min(overlap_score + complementarity_score + boost, 1.0)


class SynergyNet:
    """L3 stand-in: scores a whole regimen as the mean of its pairwise scores."""
    def score(self, regimen):
        if len(regimen) < 2:
            return {"mean_pairwise_synergy": 0.0, "pairs": {}}
        pairs = {}
        for a, b in itertools.combinations(regimen, 2):
            pairs[(a, b)] = pairwise_synergy(a, b)
        return {"mean_pairwise_synergy": sum(pairs.values()) / len(pairs), "pairs": pairs}


# ---- HCOS: holistic combination objective score ----
def toxicity_overlap_penalty(regimen):
    """Counts shared toxicity categories across any two drugs in the regimen."""
    penalty = 0
    for a, b in itertools.combinations(regimen, 2):
        tox_a, tox_b = DRUG_TOXICITY.get(a, {}), DRUG_TOXICITY.get(b, {})
        shared = [c for c in tox_a if tox_a.get(c) and tox_b.get(c)]
        penalty += len(shared)
    return penalty


def evidence_strength(regimen):
    """Simple stand-in for KG++ evidence lookup: known trial-backed pairs score higher."""
    score = 0.0
    for a, b in itertools.combinations(regimen, 2):
        if frozenset([a, b]) in KNOWN_SYNERGY_BOOST:
            score += 0.3
    return min(score, 1.0)


def mechanistic_diversity_penalty(regimen):
    """Penalize regimens that redundantly re-target the same gene without cause."""
    genes = [DRUG_TO_GENE.get(d) for d in regimen]
    return len(genes) - len(set(genes))


def HCOS(regimen, local_scores):
    synergy = local_scores["mean_pairwise_synergy"]
    evidence = evidence_strength(regimen)
    tox_penalty = toxicity_overlap_penalty(regimen) * 0.2
    diversity_penalty = mechanistic_diversity_penalty(regimen) * 0.3
    size_bonus = 0.05 * (len(regimen) - 2) if len(regimen) > 2 else 0.0  # mild reward for reaching regimens, not just pairs
    return synergy + evidence + size_bonus - tox_penalty - diversity_penalty


# ---- possible extensions: only actual drugs for the patient's altered genes ----
class DrugGraph:
    def __init__(self, available_drugs):
        self.available_drugs = available_drugs

    def possible_extensions(self, regimen):
        return [d for d in self.available_drugs if d not in regimen]


# ---- MDCOE beam search (the pseudo-code from the report, implemented for real) ----
def MDCOE(drug_graph, synergy_net, hcos_fn, beam_width=50, max_depth=5, top_k=200):
    """
    BUGFIX (see chat): the original version only returned regimens of
    EXACTLY max_depth size, because it kept extending every beam member
    forward each iteration and only looked at the final depth's survivors
    at the end. A high-scoring 3-drug regimen could exist, get extended
    into worse-scoring 4- and 5-drug versions, and never be reported --
    even though the proposal's stated top result (afatinib + alpelisib +
    trastuzumab, HCOS=0.450) IS a 3-drug regimen, and searching with
    max_depth=5 as originally written could never surface it.

    Fix: record every candidate regimen (length >= 2) generated at EVERY
    depth into a single pool, not just the ones that survive beam pruning
    into the next depth's expansion. The final ranking is taken across
    that whole pool, so the true best-scoring regimen at ANY size from 2
    up to max_depth is found automatically -- matching the proposal's
    "2-5-drug regimens" search description, rather than requiring the
    caller to already know and manually set the right max_depth.
    """
    beam = [[]]
    best_by_key = {}  # frozenset(regimen) -> (regimen, score), across ALL depths >= 2

    for depth in range(1, max_depth + 1):
        candidates = []
        seen = set()
        for regimen in beam:
            for drug in drug_graph.possible_extensions(regimen):
                new_regimen = regimen + [drug]
                key = frozenset(new_regimen)
                if key in seen:
                    continue  # regimen order doesn't matter; avoid counting permutations as distinct
                seen.add(key)
                local_scores = synergy_net.score(new_regimen)
                score = hcos_fn(new_regimen, local_scores)
                candidates.append((new_regimen, score))

                if len(new_regimen) >= 2:
                    if key not in best_by_key or score > best_by_key[key][1]:
                        best_by_key[key] = (new_regimen, score)

        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = [r for r, s in candidates[:beam_width]]
        if not beam or all(len(r) == 0 for r in beam):
            break

    final = sorted(best_by_key.values(), key=lambda x: x[1], reverse=True)
    return final[:top_k]


if __name__ == "__main__":
    # Real patient TCGA-A8-A08B: PIK3CA + BRCA2 (only 3 drugs total available — small space)
    patient_genes = ["PIK3CA", "BRCA2"]
    available_drugs = sorted({d for g in patient_genes for d in GENE_DRUGS.get(g, [])})
    print(f"=== Real patient TCGA-A8-A08B: {patient_genes} ===")
    print(f"Available drugs: {available_drugs}\n")

    graph = DrugGraph(available_drugs)
    net = SynergyNet()
    results = MDCOE(graph, net, HCOS, beam_width=20, max_depth=3, top_k=10)

    print(f"Top {len(results)} unique regimens:\n")
    for regimen, score in results:
        print(f"  HCOS={score:.3f}  {' + '.join(sorted(regimen))}")

    # Richer case: demo 6-gene patient — larger drug space, tests real pruning behaviour
    print("\n\n=== Richer test: demo 6-gene patient (EGFR, MET, SRC, BRCA1, PIK3CA, CDK4) ===")
    rich_genes = ["EGFR", "MET", "SRC", "BRCA1", "PIK3CA", "CDK4"]
    rich_drugs = sorted({d for g in rich_genes for d in GENE_DRUGS.get(g, [])})
    print(f"Available drugs ({len(rich_drugs)} total): {rich_drugs}\n")

    rich_graph = DrugGraph(rich_drugs)
    rich_results = MDCOE(rich_graph, net, HCOS, beam_width=30, max_depth=4, top_k=10)

    print(f"Top {len(rich_results)} unique regimens (beam search pruned a much larger space):\n")
    for regimen, score in rich_results:
        print(f"  HCOS={score:.3f}  {' + '.join(sorted(regimen))}")
