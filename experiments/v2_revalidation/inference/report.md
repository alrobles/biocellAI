# R6-INFERENCE — paired comparisons and automated gates

Seeds 0/1/2, donor-level held-out folds, both pathology cohorts.
Statistic: per-seed paired Spearman Δρ, pooled across seeds; donor-clustered
bootstrap (donor = resampling unit, 2000 valid replicates each).
Permutation nulls: 300 shared train-donor→label maps per cohort (100 per
seed): B2 retrains on permuted labels (N1 supervision null); other arms
permute the donor-ridge target on frozen features.

## Gates (gates.json)

| gate | decision |
|---|---|
| integrity (fold sha256 recorded = file) | PASS |
| improvement T2/T3 vs B2 (Δρ ≥ 0.05, CI_low > 0) | FAIL |
| semantics T2/T3 vs T1 | FAIL |
| semantics vs readout-perm null (ρ > q95) | FAIL (ρ 0.23/0.22 ≤ q95 0.40/0.44) |
| historical G5b (ρ ≥ 0.5) | FAIL |
| transfer strict B2 (ρ ≥ 0.35, all seeds) | PASS both directions |
| composition (state block over cov+comp, CI > 0) | FAIL |
| negative_result_valid (matrix complete) | PASS |

## Headline pooled deltas

- SEA-AD T2−B2 Δρ = −0.30 [−0.67, 0.10]; T3−B2 Δρ = −0.31 [−0.67, 0.09]
- ROSMAP T2−B2 Δρ = −0.38 [−0.62, −0.13] (FDR q < 0.05 — text arm
  significantly **below** the supervised baseline)
- B2−B0: +0.38 SEA-AD (ns), +0.34 ROSMAP [0.03, 0.62]
- B2−B1 ≈ 0 both cohorts — pseudobulk matches the supervised MLP on
  donor pathology ranking
- N1 supervision null retains ρ ≈ 0.29/0.38 (mean over 300 replicates) —
  residual donor-level structure passes through the encoder; the earlier
  single-permutation N1 was not a plumbing artifact
- Text arms do not exceed their own frozen-encoder permutation nulls

## Interpretation

Under the corrected protocol, equivalent non-text supervision is not
improved upon by caption grounding; on ROSMAP it is significantly worse.
The protocol registers this as a valid negative result. The historical
G5b rho ≥ 0.5 claim does not reproduce.

The one positive gate is strict cross-cohort transfer of the supervised
predictor (B2 ≥ 0.46–0.50 both directions), now evaluated as its own
claim independent of text grounding.

Files: paired_comparisons.csv (pooled deltas + CIs + empirical p + FDR q),
perm_nulls_long.csv (4800 null replicates), gates.json, manifest.json.
