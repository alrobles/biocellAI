# R2-TRANSFER — strict cross-cohort pathology transfer

Protocol per ADR-011: **strict** mode fixes encoder, normalization,
axis/predictor, and sign in the source cohort — zero target fitting and zero
target labels. **refit** mode keeps the source features but re-orients the
donor readout on target *train* donors (supervised orientation; the separate
calibrated analysis) and is evaluated on target *test* donors.

Method: `scripts/v2_transfer.py` + `src/biocellai/transfer.py`. Target
expression is recomputed from each cohort's verified-counts extract with the
same per-cell transform as the source fold (total-1e4 → log1p), projected onto
the source's 2000-gene space; cell universe = the frozen target fold.
The donor readout (StandardScaler+Ridge, α=1.0) is refit on source train
donors — verified to reproduce each run's held-out predictions to
≤1.1e-6 max abs diff. Gates: strict rho ≥ 0.35.

Runs: 2 directions × seeds 0/1/2 × 9 arms. Outputs per
`<direction>/s<seed>/`: `metrics.csv`, `predictions_<arm>_<mode>.csv`,
`manifest.json`. Aggregates: `metrics_long.csv`, `strict_rho_all_donors.csv`,
`refit_rho_test_donors.csv`.

## Strict transfer — Spearman rho on all target donors

| arm | rosmap→seaad s0 | s1 | s2 | seaad→rosmap s0 | s1 | s2 |
|---|---|---|---|---|---|---|
| B0 | −0.239 | −0.239 | −0.239 | 0.052 | 0.052 | −0.052 |
| B1 | 0.411 | 0.397 | 0.343 | 0.330 | 0.325 | 0.297 |
| B2 | **0.498** | **0.475** | **0.463** | **0.493** | **0.438** | **0.460** |
| N1 | 0.242 | 0.217 | 0.013 | 0.039 | 0.139 | 0.157 |
| T0–T3 | ≤0.09, mostly negative | | | ≤0.13 | | |
| N2 | ≤0.10 | | | ≤0.10 | | |

(n = 84 SEA-AD donors / 111 ROSMAP donors, all with valid pathology)

## Findings

- **B2 (supervised dual-head MLP) passes the 0.35 transfer gate in both
  directions on all 3 seeds** — the first strict, fully source-fitted
  cross-cohort result. Pseudobulk (B1) transfers partially
  (0.30–0.41, below gate in 5/6 runs).
- Text-grounded encoders (T0–T3) do **not** transfer better than supervised:
  strict rho ≤ 0.13 and frequently negative. The historical M11-A2
  caption-transfer figure (~0.478) is **not** reproduced under the strict
  protocol — that configuration differed (label-assigned pathology captions).
- N1 shows residual transfer up to 0.24 — encoder trained on permuted labels
  still carries weak structure (plausibly composition-mediated); recorded for
  R6-INFERENCE paired null analysis.
- B0 is not evaluable across cohorts: cell-type vocabularies are disjoint
  (label coverage 2.6% SEA-AD / 5.3% ROSMAP), so composition fractions
  collapse to ~constant. Recorded honestly rather than dropped.
- Refit (orientation) rows in `refit_rho_test_donors.csv` are a separate
  analysis: once the readout sees target train labels, most arms reach
  0.2–0.55 — feature geometry transfers better than the frozen axis.

## Integrity

- `gene_coverage` = 1.0 in all 6 runs (no zero-fill needed when going through
  raw extracts).
- Refit-vs-run verification ≤ 1.1e-6 max |Δpred| on source test donors for
  every arm/seed.
- Per-donor predictions exported for R6 paired inference.

## Deviations / limitations

- B0 composition vocabulary does not transfer between annotation schemes —
  reported as low-coverage degenerate output, not dropped.
- `refit` evaluates on target test donors only (train donors used for
  orientation); strict evaluates all donors plus the test-only subset.
- Cross-cohort donor overlap between SEA-AD and ROSMAP is unknown (different
  donor universes assumed; no shared-donor audit possible from available
  metadata).
