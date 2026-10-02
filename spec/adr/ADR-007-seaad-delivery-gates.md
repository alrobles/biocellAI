# ADR-007: SEA-AD delivery gates — CellOLMo deliverables → measurable gates

- Status: accepted (user-approved decisions 2026)
- Supersedes: none; extends ADR-005 direction

## Context

The Allen Institute CellOLMo role specifies eight deliverables. This ADR
converts each into a measurable, falsifiable gate so the paper reports
*decisions against pre-registered thresholds*, not vibes.

## Approved scope decisions

- **SEA-AD scope**: MTG (~84 donors, primary focus) **+ V1C** as the
  resistant-region control — enables the regional-vulnerability gate
  without the full multi-region cost.
- **Progression axis**: primary target = quantitative neuropathology
  (pTau AT8 % area / 6E10 Aβ % area, continuous); validation = Braak
  stage + CERAD ordinal agreement.
- **scFM reproduction**: all four (scGPT, Geneformer, CellWhisperer,
  C2S-Scale) — benchmark table on identical held-out-donor splits.
- **OLMo**: enters now as the text encoder (swap MiniLM/SapBERT →
  OLMo-2 family embeddings), satisfying the "open model" deliverable
  in the minimal honest way before any post-training.

## Gates

| Gate | Deliverable | Metric | Pass threshold |
|------|-------------|--------|----------------|
| G1 | OLMo multimodal model | zeroshot F1, matched register | within 0.03 of best non-OLMo arm; weights+config released |
| G2 | Reproduce 4 scFMs | benchmark table: zeroshot/probe F1 + marker-recovery@k on identical splits | table complete; report vs published claims |
| G3 | Adapt to SEA-AD | pipeline runs end-to-end, per-type cell counts ≥ M3 | completes |
| G4 | Controlled grounding exp. | same arm matrix as M3–M6 on SEA-AD | all arms × 3 seeds complete |
| G5a | Donor-level representation | Spearman ρ(donor-embedding, AT8%/6E10%) on held-out donors | grounded > cell-only pseudobulk baseline, ρ ≥ 0.5 |
| G5b | Disease progression | rank-corr of embedding-trajectory order vs Braak/CERAD | ρ ≥ 0.5 held-out |
| G5c | Regional vulnerability | MTG vs V1C: predicted vulnerability ranking agrees with known tau topography (V1C spared) | directional agreement held-out |
| G6a | Cell-type recovery | zeroshot F1 held-out donors | ≥ supervised-relative performance of M3 |
| G6b | Marker recovery | precision@k of predicted markers vs DE/literature markers | ≥ 0.6 @ k=10 |
| G6c | Vulnerability agreement | rank-corr predicted vs published vulnerable cell types (L2/3 IT, SST, etc.) | ≥ 0.6 |
| G6d | Expert review | plain-language Q&A rubric + inter-rater κ | protocol + rubric defined; score reported |
| G7 | Allen iteration | predictions list with per-prediction confidence | artifact delivered |
| G8 | Open release | data manifests, code, weights (HF), paper tables | checklist complete |

## MVP plan (M7)

1. Census query: SEA-AD MTG + V1C, primary data only; stratified
   subsample ~250–300k nuclei → `/beegfs/.../seaad_mtg_v1c.h5ad`.
2. Verify pathology columns in census `obs` (Braak/CERAD/Thal +
   quantitative AT8/6E10 fields); fall back to Allen donor-metadata
   table (Sage Synapse / AWS manifest) joined by donor_id.
3. Brain markers: PanglaoDB brain cell types + CellMarker; retrieval →
   Qwen synthesis pipeline reused unchanged (`type_llm` contract).
4. Donor embedding = mean-pooled cell embeddings per donor; regress on
   AT8%/6E10% (ridge), held-out donors; baseline = pseudobulk-PCA donor
   embedding.
5. Trajectory: principal curve / diffusion pseudotime over donor
   embeddings → rank-corr vs Braak ordering.

## Honest-risk notes

- Census `obs` may carry only ordinal pathology (Braak/CERAD); the
  quantitative AT8/6E10 fields live in the donor-level supplement —
  join step required, keep provenance.
- 84 donors total → held-out regression has ~20 donors; ρ estimates are
  noisy. Pre-register the comparison as grounded vs cell-only, not an
  absolute ρ claim.
- scFM reproduction: C2S-Scale is Gemma-2-27B (~54 GB bf16) — run
  quantized or on A100-80G; if infeasible, document partial repro
  explicitly rather than dropping silently.
