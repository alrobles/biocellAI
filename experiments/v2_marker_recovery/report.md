# R5-MARKERS — non-circular marker recovery report

Evaluates whether the fold-specific, train-only Wilcoxon marker ranking
(the same pipeline that injects the top-8 genes into T2 marker captions)
recovers canonical marker genes, per ADR-011: *"separate genes used in
captions from independent evaluation references; retain the precision@10
≥ 0.6 threshold."*

## Design (non-circular)

- **Candidates**: full Wilcoxon ranking on train cells only (same call as
  `scripts/v2_captions.marker_captions`), minus the genes injected into
  the fold's own marker captions → next-10 candidates.
- **References (external, label-free)**:
  - blood → `data/manifests/panglao_markers.json` (CellOntology map)
  - seaad_mtg/seaad_pfc → `data/manifests/panglao_brain_markers.json`
  - rosmap → `data/manifests/panglao_rosmap_markers.json` (derived
    here from the same canonical sets; `scripts/build_rosmap_marker_ref.py`)
- **Metric**: P@10 macro over evaluable classes (non-empty reference),
  averaged over seeds per cohort. Gate: ≥ 0.6.
- Folds evaluated: revalidation s0-2 (blood, seaad_mtg, rosmap) +
  confirmation (seaad_mtg s3-5, rosmap s3-5, seaad_pfc s0-2). No fold
  was evaluated before this task — markers/captions are fold-specific.

## Results

| cohort | seeds | evaluable classes | P@10 non-circular | P@10 naive | ceiling* |
|---|---|---|---|---|---|
| blood | 0-2 | 16/16 | **0.019** | 0.131 | 0.156 |
| seaad_mtg | 0-5 | 22/24 | **0.054** | 0.051 | 0.382 |
| rosmap | 0-5 | 32/34 | **0.020** | 0.032 | 0.206 |
| seaad_pfc | 0-2 | 22/24 | **0.024** | 0.050 | 0.259 |

\* mean per-class ceiling = min(10, n_reference ∩ HVG space)/10 — the
maximum reachable P@10 given the fold's 2000-gene HVG space.

**Gate decision: FAIL for all four cohorts — and unreachable by
construction.** Only 13-33% of each reference's canonical genes are even
present in the fold HVG space, so the mean ceiling (0.16-0.38) is below
0.6 before any marker quality is tested.

## Interpretation

Two distinct findings, both honest:

1. **The P@10 ≥ 0.6 gate is structurally unreachable in this protocol.**
   It presumes canonical markers live in the evaluation space; the
   HVG-2000 selection excludes most of them (blood: 25/192 reference
   genes in space). A variant scored over the full gene set or against
   an HVG-restricted reference would be needed for the gate to be
   meaningful — that variant was not pre-registered, so we report the
   gate as FAIL rather than silently redefining it.

2. **Within the reachable ceiling, recovery is also low** (~2-5% of the
   possible ~15-38%). The fold-derived top markers are subtype-resolved
   DE genes (e.g., `CBLN2`, `THSD7A` for cortical subtypes), while the
   PanglaoDB brain reference is category-coarse — all 9 excitatory
   subclasses share one 12-gene generic set (`ENO2`, `SNAP25`, `NEFM`…).
   Genuine hits concentrate where vocabularies align (GABAergic `GAD2`/
   `SLC6A1`, oligodendrocyte `ENPP2`/`TF`, astrocyte `SLC1A3`), consistent
   with the M7 caveat that PanglaoDB resolves only coarse neural
   categories. Subtype-level validation needs a subtype-level external
   reference (e.g., the Allen Brain Atlas marker taxonomy) — flagged as
   future work, not silently substituted.

Non-circularity verified: naive P@10 (caption genes included) ≈
non-circular — the injected caption genes were not the ones hitting the
reference anyway.

## Provenance

- `src/biocellai/marker_eval.py`, `scripts/v2_marker_recovery.py`,
  `scripts/slurm/v2_marker_recovery.sbatch`; SLURM jobs 30847130-33.
- Outputs: `{cohort}_per_seed.csv`, `{cohort}_classes.json` (candidates,
  hits, per-class ceiling), `{cohort}_manifest.json` (input SHA-256,
  gate).
- ROSMAP reference derived from the same PanglaoDB canonical sets as the
  brain map + the blood T-cell set; CPEC/Epd have no PanglaoDB category
  and are excluded (evaluable counts reflect that).
