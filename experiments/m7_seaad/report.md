# M7 — SEA-AD brain grounding + disease-progression evaluation

**Status: complete (first pass).** Gates G5a/G5b not met; G5c shows region-specific signal. New leakage channel identified and controlled: LLM parametric marker knowledge.

## Data layer

- Source: official SEA-AD AWS release `2026-06-22` (not CELLxGENE — census obs lacks pathology columns).
- `seaad_mtg_v1c_240026_s0.h5ad`: 240,026 nuclei, stratified by (donor × Subclass).
  - MTG: 120,034 cells, 84 donors, 24 subclasses.
  - V1C: 119,992 cells, 43 donors, 21 subclasses.
- `data/seaad_donors.csv`: 127 donor×region rows with Braak/CERAD/Thal/ADNC/CASI/APOE + Allen **Continuous Pseudo-progression Score** (global + per-region, Aβ/pTau components) joined from `Global_and_Local_CPS.20260105.csv`.
- `data/manifests/markers_seaad_s0.json`: train-donor marker table (8 markers/subclass, seed 0).
- `data/manifests/panglao_brain_markers.json`: curated PanglaoDB map — coarse neural categories only; **all glutamatergic cortical subclasses collapse to one marker set** (ENO2/BEX2/BEX1/SNAP25/VSNL1/DYNC1I1...).

Build memory fix: anndata backed fancy-indexing materializes the whole CSR (MTG = 6.6B nnz float64 ≈ 105GB → OOM at 128G and 224G). `_subset_backed_csr()` gathers only selected rows via h5py indptr arithmetic. Job `29922891` completed in 67 min at 130GB peak.

## Retrieval (brain context, `NEURAL_CONTEXT` prior)

72 type×arm rows in `m7_retrieval/retrieval_metrics.csv`. Captions written to `data/text/m7/` (M6 captions live in `data/text/` — per-dataset `textdir` now avoids the collision that overwrote the blood files).

## Grounding (donor-held-out, 3 seeds, 24 subclasses)

| arm | source | matched F1 | marker-reg F1 |
|---|---|---|---|
| cell_only_supervised | — | 0.989 | — |
| rag_markers_alias | train markers → PubMed prose | 0.887 | 0.17 |
| rag_curated_alias | PanglaoDB brain → PubMed prose | 0.233 | 0.19 |
| olmo_honest_markers | train markers → docs → OLMo-2-13B | 0.989 | 0.35-0.46 |
| olmo_honest_curated | PanglaoDB → docs → OLMo-2-13B | 0.989 | 0.45 |
| olmo_labeled | name + docs → OLMo (upper bound) | 0.989 | 0.93 |
| prior_curated_olmo | generic markers, **no excerpts** | 0.945 | 0.22 |
| prior_markers_olmo | train markers, **no excerpts** | 0.988 | 0.70 |

### Key findings

1. **Curated brain markers actively hurt**: identical captions for ~10 glutamatergic subclasses → InfoNCE merges them → probe collapses to 0.25 (vs 0.95 for other arms). Coarse external knowledge is degenerate at fine granularity — the predicted failure mode, confirmed.
2. **Matched-register eval saturates**: ~0.94-0.99 for every LLM-synthesized arm, including prior-only controls. Sampling noise alone produces discriminative nearest-caption classification. Marker-register and probe remain informative; matched F1 should not be reported as evidence at this granularity.
3. **LLM parametric knowledge is a real leakage channel**: with identical retrieval docs and identical generic markers, OLMo named the *correct canonical layer markers* (CUX2 for L2/3 IT, RORB for L4 IT). The prior-only control (`--prior-only`) isolates this: 0.22 marker-reg F1 for generic markers vs 0.45 with excerpts — **retrieval contributes +0.23** even when pooled docs are subclass-identical (excerpts mention layer-specific markers OLMo extracts).
4. Leakage flags (literal type-name in honest output): 2/24 (markers), 3/24 (curated) — recorded in `*.prov.json`.

## Progression (G5a/G5b/G5c — held-out donors, ridge on donor-mean embeddings)

| arm | CPS_Global | CPS_Local@MTG | Braak trajectory |
|---|---|---|---|
| pseudobulk PCA | 0.214 | — | 0.319 |
| rag_markers_alias | 0.16-0.21 | 0.29-0.36 | −0.13 |
| rag_curated_alias | 0.26 | 0.42-0.43 | −0.32 |
| olmo_honest_markers | 0.29-0.30 | 0.34-0.35 | ±unstable |
| olmo_honest_curated | 0.31 | 0.39-0.41 | ±unstable |
| prior_markers_olmo | 0.27-0.31 | 0.33-0.36 | **+0.39-0.40 (both encoders)** |

- **G5a: not met** (grounded ρ≈0.30-0.43 vs gate 0.50; all grounded arms do beat pseudobulk's 0.21 on CPS).
- **G5b: not met** — only `prior_markers` has a stable positive Braak trajectory (+0.40); other arms flip sign between encoders (noise).
- **G5c**: progression signal is region-specific — MTG ρ≈0.35-0.43, V1C ≈0-0.2 — consistent with MTG's higher AD vulnerability.

Interpretation: a cell-type-identity contrastive objective encodes pathology only weakly at the donor level. Donor-level progression likely needs a donor/patient-level objective (e.g. pathology-conditioned captions or donor-level contrastive) rather than type-level grounding.

## Provenance

- Jobs: build `29922891`, retrieval `29922900`, grounding `29922904/905` + `29927269/70/71`, prior-only `29987788-791`, progression `29927250/51`, `29927260/61`, `29927267/68`, `30012368-372`.
- Cell embeddings (1.9GB/arm): `experiments/m7_*/embeddings/` on beegfs — not committed (gitignored); regenerate via `scripts/slurm/m7_seaad_grounding.sbatch`.
