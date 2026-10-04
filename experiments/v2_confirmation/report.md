# R6-CONFIRMATION — confirmatory evaluation report

Confirmatory replication of the v2 revalidation results on never-inspected
evaluation sets, under the frozen protocol from `spec/confirmation_plan.md`.
Exploratory = `experiments/v2_revalidation` (MTG + ROSMAP, seeds 0–2).
Confirmatory = `experiments/v2_confirmation` (MTG + ROSMAP, seeds 3–5;
SEA-AD PFC, seeds 0–2).

## Design

- Fresh donor splits: MTG and ROSMAP seeds 3–5 (donors overlap with
  exploratory but the splits were never inspected before this run).
- New region: SEA-AD PFC (120k cells, 80 donors, seeds 0–2). Donor
  overlap with MTG is documented — same-study donors, disjoint cells.
- Protocol frozen before any confirmatory metric was computed: same code
  (`main` @ 33c838f+), same arm set (B0/B1/B2/T0–T3/N1/N2), same gates.
- 100 shared donor→label permutation maps per cohort × seed (900 total);
  donor-clustered paired bootstrap (2000 replicates); BH-FDR on
  secondary comparisons.

## Gate decisions (automatic)

| Gate | Result |
|---|---|
| integrity (fold SHA, matrix completeness) | **PASS** |
| improvement T2 vs B2 | FAIL (rho 0.261 ≤ null q95 0.418) |
| improvement T3 vs B2 | FAIL (rho 0.228 ≤ null q95 0.456) |
| semantics T2/T3 vs T1 | FAIL |
| semantics T2/T3 vs null | FAIL |
| historical G5b (rho ≥ 0.5) | FAIL |
| transfer strict (≥ 0.35) | **PASS — all 3 directions, all seeds** |
| composition | NOT_EVALUABLE (state-analysis not in scope) |
| negative_result_valid | **PASS** |

## Headline result: strict transfer replicates

| Direction | Seeds (strict rho, all_donors) | Gate |
|---|---|---|
| rosmap → seaad_mtg | 0.472 / 0.526 / 0.475 | PASS (≥0.35) |
| seaad_mtg → rosmap | 0.400 / 0.460 / 0.497 | PASS (≥0.35) |
| seaad_mtg → seaad_pfc | 0.775 / 0.764 / 0.785 | PASS (≥0.35) |

MTG→PFC transfer is substantially higher than the cross-study legs —
consistent with same-study donors and shared preprocessing, not with
independent cohort evidence. It is reported as a robustness check, not
as an independent replication.

## Paired comparisons (donor-clustered bootstrap, 2000 reps)

**SEA-AD MTG (s3–5, n=45 test donors):**
- T2 − B2 = −0.19, CI [−0.49, 0.11]
- T3 − B2 = −0.23, CI [−0.49, 0.05]
- T2 − T1 = +0.04, CI [−0.04, 0.13]
- B2 − B0 = +0.19, CI [−0.13, 0.50]; B2 − B1 = +0.08, CI [−0.16, 0.33]

**ROSMAP (s3–5, n=68):**
- T2 − B2 = −0.27, CI [−0.48, −0.05], q=0.093
- T3 − B2 = −0.29, CI [−0.52, −0.05], q=0.093 — text significantly *below* B2
- B2 − B0 = +0.21, CI [−0.03, 0.44]

**SEA-AD PFC (s0–2, n=46):**
- T2 − B2 = −0.11, CI [−0.43, 0.21]
- T3 − B2 = −0.06, CI [−0.34, 0.24]
- N1 − B2 = −0.27, CI [−0.54, 0.04]

## Interpretation

The confirmatory runs reproduce the exploratory picture:

1. **Text arms do not beat supervised baselines.** T2/T3 sit at or below
   T1 (random prototypes) and well below B2, replicating the exploratory
   negative result on unseen splits and a new region.
2. **B2 strict cross-cohort transfer is the robust positive claim.**
   Donor-level pathology ranking transfers between MTG and ROSMAP in
   both directions on never-inspected seeds, and MTG→PFC transfer is
   even stronger (with the same-study caveat).
3. **N1 retains residual signal** (rho 0.36/0.41 on MTG/ROSMAP) —
   the donor-structure/composition confound persists under
   confirmation, reinforcing that composition control is required.
4. Per-seed deltas are noisy (e.g., MTG T2−B2: +0.10, −0.43, −0.25
   across seeds 3/4/5) — single-seed readouts are unreliable, which
   the donor-clustered bootstrap across seeds handles correctly.

## Provenance

- Runs: `experiments/v2_confirmation/{cohort}/run_s*_full/` (metrics.csv,
  manifest.json, predictions_*.csv per arm)
- Permutation nulls: `*/perm_null_s*/perm_{0..99}.csv` (900 files,
  300/arm/cohort — N1 is itself the label-permuted arm and has no null)
- Inference: `inference/paired_comparisons.csv`, `inference/gates.json`,
  `inference/manifest.json`
- Transfers: `transfer/{direction}/s*/metrics.csv` + strict/refit
  predictions
- HPC chain: prep 30798624 → captions 30798625 → runs 30798627-29 →
  nulls (30834427-33 + 30799068-70) → transfers (30799071-73, 30834467-68)
  → inference 30834472
- Data fix: `seaad_donors_pfc.csv` CPS_Global filled from donor-level
  allregions table (commit 33c838f); PFC folds rebuilt after the fix.
