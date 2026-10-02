# ADR-008: M11 improvement roadmap — attack the held-out pathology score

Status: proposed (2026-09-23)

## Context

Baseline (M10 dual-head, SEA-AD all-regions, donor-held-out, 3 seeds):
`dual_cog` CPS_Global ρ = 0.56–0.60, ADNC/CERAD trajectories 0.70–0.72,
cell-type zero-shot 0.89–0.91 balacc. External scFMs on the identical split
(G2): scGPT CPS ρ 0.43–0.51, Geneformer V2 0.13–0.46, V1 0.28–0.37.

Structural bottleneck identified in M9: **84 unique donors → ~21 held-out
per split**. Spearman ρ at n=21 has a wide CI; cells-per-donor and regions
are already maxed out inside SEA-AD. Raising the score materially requires
attacking (a) donor count, (b) donor-level signal extraction, or
(c) objective granularity — not more cells.

## Decision

Three parallel options, ordered by expected gain-per-cost. All reuse the
existing donor-held-out protocol, shuffle-null controls, and pre-registered
gates (Δρ ≥ +0.05 vs same-split baseline AND survives label permutation).

### Option A — More donors: ROSMAP replication (highest impact)

ROSMAP snRNA-seq (Mathys lab, Nature 2024): ~427 donors, 6 brain regions,
~2.3M nuclei, with braaksc/ceradsc/cogdx + continuous amyloid & tau burden.

- **A1 — Cohort replication**: build `rosmap_s0.h5ad` + donor table
  (target = continuous tau/amyloid burden or cogdx; SEA-AD CPS_Global has
  no direct analog — map carefully, document), retrain `dual_cog` →
  n_test ~100 donors, CI tightens ~5x.
- **A2 — Zero-shot cross-cohort transfer**: apply the SEA-AD-trained M10
  encoder to ROSMAP cells/donors without retraining (and vice versa).
  ρ ≥ 0.35 in the transfer direction = external-validity claim, the
  strongest possible evidence that the objective captures biology rather
  than cohort idiosyncrasy.

Cost: Synapse download (~50–100 GB), build ~2h, reruns ~30min.
Fallback: if usable donors < 200 or metadata mapping fails, document and
drop to Option-B-only.

### Option B — Architecture upgrades (cheap, no new data)

- **B1 — MIL/attention donor pooling**: replace mean-pool with a learned
  attention pooler over cells; donor signal (~7% of variance) likely lives
  in subpopulations (reactive glia, Sncg) that mean-pooling dilutes.
  ~1 day incl. reruns.
- **B2 — Soft ordinal contrastive**: weight the pathology InfoNCE by
  |target_i − target_j| (continuous CPS/tau), recovering the ordinal axis
  the binary `cog` caption discards. ~1 day.
- **B3 — SSL pretrain → pathology head**: SimCLR-style contrastive on raw
  expression over all 440k nuclei (unlabeled), then attach pathology head.
  Uses data the text objective ignores. ~2 days; optional.

### Option C — Corpus grounding upgrade

- **C1 — Retrieved pathology captions**: reuse the M6 PubMed retrieval
  pipeline to generate per-level pathology captions (braak/cerad/adnc/
  cognition) from literature instead of hand-written dictionaries →
  answers the "caption engineering" critique and is a robustness control.
  ~half day (pipeline exists).
- **C2 — large-corpus pretraining (CELLxGENE/Genecorpus-30M)**: deferred —
  reproduces an scFM, high compute, low marginal value for the thesis.

## Route (phases)

- **P0 (done)**: G2 same-split table — Geneformer V1/V2, scGPT, CellWhisperer.
- **P1 (parallel)**:
  - A-data: locate + download ROSMAP, build h5ad + donor table.
  - B1: MIL pooling retrain on SEA-AD `dual_cog`.
  - C1: retrieve pathology captions (m6 pipeline).
- **P2**: SEA-AD micro-sweep: dual_cog × {B1, B2, C1} factorial, same
  splits/seeds/gates; winner replicated on ROSMAP (A1).
- **P3**: A2 cross-cohort transfer both directions; bootstrap CIs on all
  headline numbers.
- **P4**: report + paper update; remaining gates G6b (markers), G6d
  (expert review), G8 (open release).

## Gates / success criteria (pre-registered)

| Claim | Gate |
|---|---|
| Improvement over M10 baseline | Δρ ≥ +0.05 same-split, survives shuffle-null |
| ROSMAP replication | CPS-analog ρ ≥ 0.5 at n_test ~100 |
| Cross-cohort transfer | ρ ≥ 0.35 zero-shot direction |
| Honest reporting | All runs vs identical splits; tokenizer/version provenance logged |

## Risks

- ROSMAP metadata differs (no CPS_Global) → define tau/amyloid composite
  target before training; document as protocol difference.
- Cross-cohort gene space / cell-type vocab mismatch → intersect on
  symbols; report coverage.
- B2 soft-label leakage: target is donor-level, split stays donor-level.
