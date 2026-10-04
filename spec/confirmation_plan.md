# R6-CONFIRMATION — exploratory vs confirmatory evaluation

Per ADR-011 R6: seeds 0/1/2 are **exploratory** (prior reuse). Confirmation
requires a genuinely uninspected evaluation under the frozen v2 protocol.

## Status of existing results

All artifacts under `experiments/v2_revalidation/` (runs s0–s2, transfer,
perm nulls, inference, state_analysis) are **exploratory**. No pipeline
decision, hyperparameter, or gate threshold changes after this freeze.

## Confirmation design (protocol frozen at this commit)

| set | cohort | seeds | what is uninspected |
|---|---|---|---|
| A | seaad_mtg | 3,4,5 | donor splits (new partition; same donors) |
| A | rosmap | 3,4,5 | donor splits (new partition; same donors) |
| B | seaad_pfc | 0,1,2 | new region/cells; **donor IDs overlap MTG** — partial independence only |
| transfer | rosmap↔seaad_mtg | 3,4,5 | strict transfer on new splits |
| transfer | seaad_mtg→seaad_pfc | 0 | strict transfer to never-analyzed cells |

Procedure: prep folds → captions (train-only, same modes: T2=markers,
T3=pubmed_rag copy of the registered corpus, N2 permutes T3) → full 9-arm
matrix → 100 shared perm maps per (cohort, seed) → strict transfer →
`v2_inference.py` once, blind, over `experiments/v2_confirmation/`.

No intermediate metrics are inspected; gates are evaluated once at the end.

## Limitations (must appear wherever confirmation is claimed)

- Sets A reuse the same biological donors as exploratory runs — new splits,
  not new individuals.
- Set B (PFC) shares donor IDs with MTG (multi-region donors) — cells and
  region differ, but donor-level pathology labels correlate by construction.
- Confirmation therefore measures protocol stability on unseen partitions/
  cells, not fully independent replication.

## Wording rule

Claims phrased as confirmatory (beyond "consistent with" language) are only
allowed for results covered by the confirmation sets above; everything else
is reported as exploratory.
