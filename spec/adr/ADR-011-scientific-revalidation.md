# ADR-011: Scientific revalidation and paper delivery master plan

Status: in progress — infrastructure first; v2 results are still pending.
Historical reference: commit `5d72702` (M0–M12 and the M13 composition diagnostic).
This ADR precedes new v2 runs; it does not make historical analyses confirmatory
or constitute an external preregistration record.

## Research question and scope

Under what conditions does alignment with biological text improve generalization
to unseen donors compared with equivalent non-text supervision, and how much
persists after controlling for cell composition, region, and cohort differences?

Objectives: (O1) distinguish semantics, supervision, and format; (O2) ensure
comparable data and splits; (O3) evaluate cell identity and donor state separately;
(O4) control for composition/covariates; (O5) measure inductive transfer;
(O6) reproduce scFMs using native protocols; (O7) quantify uncertainty and
biological accuracy; (O8) deliver a reproducible manuscript and software.

We do not assume that language will outperform alternatives, that composition
explains the results, or that more donors will resolve the limitation. Negative
results close tasks when the protocol and artifacts are complete.

## Non-negotiable rules

- M0–M12 are historical/exploratory results. Do not overwrite their directories.
  New outputs belong in `experiments/v2_revalidation/<run-id>/`.
- Do not use historical metrics as the baseline for a different v2 protocol:
  rerun the candidate and reference on the same cells/splits.
- A donor, not a cell or a seed, is the independent evaluation unit.
- `label-name-free`, external content, and label-free training are distinct.
  Assigning captions using `cell_type` is text-mediated supervision.
- No large GPU sweep before data validation, negative leakage tests, a one-fold
  smoke test, and review of the experiment manifest.
- Do not infer success from absence in squeue or from `pgrep`: require Slurm
  state + exit code + a complete artifact inventory + metric validation.
- Do not store secrets; the credential exposed during the previous operation
  must be rotated by its owner. Do not copy it into a manifest or the paper.
- Record any deviation from this plan before inspecting new results.

## Execution sequence and dependencies

```
R0 protocol / inventory
  -> R1 data and folds -> R2 evaluation -> R3 controls -> R6 inference
                      -> R4 scFMs --------------------> R6
R0 -> R5 biological review ----------------------------> R6
R0 -> R7 environment and provenance -------------------> R6
R6 + R5 + R7 -> R8 manuscript and release -> R9 M13 decision
```

The operational registry is `spec/revalidation_tasks.json`. Each unit has an
owner role, dependencies, a deliverable, and a completion criterion. The command
`python scripts/revalidation_status.py` shows ready/blocked tasks; it neither
submits jobs nor turns existing files into scientific evidence.

### R0 — Freeze claims and provenance (CPU/editorial)

- Inventory every number: cohort, endpoint, model, commit, job, source CSV,
  predictions, split, caption hash, and protocol. Flag missing evidence.
- Capture environments and external changes in the HPC checkout without modifying them.
- Mark the following claims as pending: label-free training, statistical
  equivalence, strict zero-shot transfer, compositional causality, scFM
  superiority, and `objective > scale`.
- Correct manuscript terminology; keep historical numbers visibly identified
  as such. The old PDF is not a submission version.

Completion: a claims registry without unsupported PASS decisions; historical
artifacts remain intact.

### R1 — Rebuild datasets and folds (CPU/HPC)

1. Inventory sources, releases, permissions, checksums, and the state of X/layers.
   Do not assume a file named raw contains counts.
2. Reconcile the 127/84 SEA-AD donors: identify neurotypical references, CPS
   eligibility, QC exclusions, and changes in the donor universe by region.
3. Validate cell/donor IDs, uniqueness, the many-to-one CPS join, and consistency
   of donor metadata. Prefix cohort identifiers before any concatenation.
4. Preserve full counts. Apply predefined cell QC to the full transcriptome;
   fit gene filters and HVGs using training data only.
5. Normalize once using the full-transcriptome total, before selecting HVGs.
   Reuse the source genes/transformation for transfer.
6. Freeze donor/cell/gene lists and hashes per fold. Controls, baselines, and
   candidates must use exactly the same evaluation universe.
7. Dataset-derived markers, retrieval, and captions must be fold-specific.
   Frozen external knowledge may be shared, with the supervision used to
   assign it to each cell explicitly stated.

Deliverables: cohort flow tables, source manifests, folds, matrices, and
versioned captions. Actual counts replace any aspirational counts.
Completion: changing test data alone does not change HVGs/markers/transformations
fitted on train; reject repeated normalization and matrices without provenance.

### R2 — Inductive evaluation and basic statistics (CPU)

- One splitter; no alternative `int`/`round` rules in baselines. NPZ files from
  trained representations retain their split and are not repartitioned for MIL.
- Fit StandardScaler and ridge on train. Fit PCA on train and apply it to test.
  If used, all-donor descriptive analyses are separate and not called held-out.
- Save per-donor predictions, observed values, split, seed, arm, target, and
  protocol; report rho, R², MAE, and effective sample size, including NaNs/failures.
- Strict transfer: fix the encoder, normalization, axis/predictor, and sign in
  the source cohort. Orientation using target labels is a separate calibrated analysis.
- Composition baseline: fractions by type and type×region, with vocabulary
  and encoding fitted on train; do not interpret recovery as absolute abundance.
  Available covariates: age, sex, PMI, region, batch/Study.

Completion: tests for separation and fit invariance to test changes; exported
predictions; no mixing of protocols within a table.

### R3 — Attribute effects to semantics versus supervision (MLP first)

Same architecture and update budget, same data, targets, and pooling. The
non-text baseline receives the same labels as the text-based model. Captions
assigned by class are not described as label-free learning.

| arm | signal | comparison enabled |
|---|---|---|
| B0 | composition + covariates | donor baseline without an encoder |
| B1 | pseudobulk, ridge/PCA | expression without encoder supervision |
| B2 | MLP with type CE + pathology CE/ordinal objective | equivalent supervision without language |
| T0 | one-hot prototypes, equal-dimensional projection | class identity without semantics |
| T1 | fixed random vectors per class | geometry/prototypes without biology |
| T2 | manual captions | manually authored biological text |
| T3 | PubMed captions, no synthesis | retrieval contribution |
| T4 | RAG captions + synthesis | synthesis contribution |
| N1 | donor→pathology permutation within train only | supervision control |
| N2 | permuted class→text mapping, consistent across train/eval | semantics versus coding |

- Predefine MiniLM/SapBERT; do not select the better one on test. Describe OLMo
  as a synthesizer unless explicitly tested as an encoder; do not claim G1 complete.
- Separate seeds for splits, initialization, prototypes, and permutations.
- Report matched-register and held-out-paraphrase evaluation; do not generate
  new evaluation captions after inspecting test errors.
- Compare embeddings with a common probe protocol and train-only tuning.
- Treat multi-positive loss for repeated captions as an explicit ablation,
  not a hidden change to the reference objective.

Completion: the minimum B0/B1/B2/T0/T1/T2/T3/N1 matrix is complete; T4 and N2
are predefined extensions. A semantic claim requires an advantage over B2 and T1,
not merely N1.

### R4 — Fair reproduction of scFMs (GPU use subject to prerequisites)

- Geneformer: official version-specific tokenizer; full counts, medians, IDs,
  special tokens, and correct sequence length. Fixture parity before jobs.
- scGPT: load the checkpoint and report missing/unexpected keys, vocabulary,
  binning, pooling, seeds, and coverage. Verify frozen→fine-tuned execution on one batch.
- CellWhisperer: pin revisions and package patches; no silent bypass of input
  checks. C2S-Scale: implement if resources/scope allow; otherwise record the
  benchmark as incomplete rather than removing it from the checklist.
- Two separate tables: native/full-transcriptome protocol and controlled HVG
  ablation. Same cells/donors and probe; disclose potential overlap between
  cohorts and pretraining data.
- For scaling, match or report updates, cells seen, LR selection, time/memory,
  and actual parameter counts. Do not extrapolate a negative result from
  3–5 epochs into a general law about capacity.

Completion: tokenizer parity, audited weights/provenance, and a complete paired
comparison. An unversioned local patch blocks the publication claim.

### R5 — Biological validation and composition–state analysis (CPU + human review)

- Two independent domain reviewers, blinded to arm performance; record
  disagreements and adjudication without inventing expert evaluations.
- Per-claim/caption rubric: lineage/function accuracy, relevance, PMID support,
  unsupported assertions, and missing necessary information. Ordinal scale 0–2;
  flag critical lineage errors separately.
- Report agreement and uncertainty; use weighted κ for ordinal rubrics where
  appropriate. Independently review captions reserved for evaluation.
- Marker recovery: separate genes used in captions from independent evaluation
  references; retain the precision@10 ≥ 0.6 threshold.
- State: centroids within type×region, pooling with fixed train-derived weights,
  and train-only residualization of composition/covariates. Use paired incremental
  comparisons; rho differences are not percentages of information or causality.
- Vulnerability: distinguish predictability, abundance changes, and the direction
  of loss/expansion; register an external biological reference before comparison.

Completion: actual expert-review tables, adjusted analyses, and non-causal wording.
If experts are unavailable, block the claim or exclude it from the paper's scope.

### R6 — Reruns, uncertainty, and gates (CPU/GPU)

- Retain seeds 0/1/2 for historical comparison, but label them exploratory due
  to prior reuse. Use nested donor-level validation for tuning and a genuinely
  uninspected cohort or evaluation for confirmation when available.
- Primary endpoint: SEA-AD CPS_Global rho. Secondary endpoints: pTau/Aβ,
  ordinals, R²/MAE, ROSMAP path_level, F1/balanced accuracy, and subgroups.
- Compare predictions on identical donors. Use paired donor-level bootstrap
  (≥2000 valid replicates); for repeated folds, preserve donor clustering rather
  than treating seed×donor pairs as independent replicates. A CI conditional on
  the fitted model does not by itself capture retraining uncertainty.
- Permutations: 100 training donor→label maps for the primary MLP matrix,
  shared across arms; a plumbing pilot is not a definitive null. Reduced budgets
  for expensive fine-tuning require correspondingly limited conclusions.
- Adjust secondary comparisons for multiplicity (FDR); do not change the primary
  endpoint or select the encoder using test. Nonsignificance is not equivalence.

| gate | required decision |
|---|---|
| Integrity | identical hashes/splits and train-only transformations; if missing, NOT_EVALUABLE |
| Improvement | exact Δrho ≥ 0.05 over the paired v2 reference and lower bound of the CI for Δ > 0 |
| Semantics | improvement over B2 and T1; survives controls evaluated against a null distribution |
| Historical G5b | rho ≥ 0.5 for the registered endpoint; a positive value is insufficient |
| Transfer | rho ≥ 0.35 with the axis/predictor fixed in the source cohort; calibrated analysis separate |
| Composition | adjusted improvement with a paired CI; no automatic causal attribution |
| Negative result | valid protocol and complete matrix; not evidence of universally absent effects |

Completion: complete artifacts, gate decisions generated without rounding, and
explicit confirmation limitations. 0.604−0.558=0.046 does not pass +0.05.

### R7 — Reproducible software, environments, and operations

- Freeze validated local/HPC environments and HF/upstream revisions; separate
  incompatible scFM-specific dependencies. Do not blindly freeze a mixed
  `pip freeze` environment or relax security policies.
- Unit tests + synthetic CPU integration + HPC smoke tests; CI must not download
  clinical datasets or large checkpoints. Record hardware version information.
- Immutable runs: manifest before work, temporary outputs and atomic publication,
  COMPLETED only after validation. Retries must not overwrite outputs.
- SLURM: CPU for construction/evaluation; GPU only for training/embeddings;
  capped arrays (pilot 1, initial maximum 2 GPU tasks), afterok dependencies.
  No heavy work on login nodes; logs on the shared filesystem, not remote /tmp polling.
- Verify packaging in a clean environment and configurable paths. Licensing and
  redistribution permissions require decisions by the owner/data custodians.

Completion: a separate installation reproduces a smoke test and tables from
artifacts; no secrets in logs; verifiable manifests/checksums.

### R8 — Manuscript, figures, and release

Final structure: motivation and estimands → data/QC → supervision and controls →
statistical protocol → identity → state/composition → transfer → scFMs → text
validation → appropriately bounded discussion → availability/limitations.

Minimum figures: data/split flow; supervision matrix; performance by source/register
with CIs; donor predictions with composition adjustment; strict/calibrated transfer;
scaling with budget and coverage. Generate tables from metrics rather than copying
them manually.

Review primary SEA-AD/ROSMAP references; do not promise released weights/data when
licenses or artifacts are missing. Recompile the PDF after tables are verified.
Completion: every claim links to evidence; abstract/discussion/conclusion are
consistent; no historical result is presented as a corrected v2 result.

### R9 — Resume M13 only after revalidation

- Open justification: more donors is a hypothesis, not an established bottleneck.
- 84+111=195 available; ~63+83=146 train under the 25% split, subject to the census.
- Expert-reviewed alias map; J0 and joint training use the same space/transformation.
- Specify whether there is 1 type head + 2 pathology heads or a shared pathology
  head with masking; freeze the choice before the sweep and balance donors too.
- J0/J1/J2/J3/J4 with comparable update budgets and cohort controls.
- Evaluating new donors from two training cohorts is not transfer to an unseen
  cohort. Reserve a leave-one-cohort-out direction or a third cohort.
- Donor-count learning curves with a fixed test set to investigate scaling.

Completion: ADR-010 revised using v2 evidence and execution authorized; no M13 job
is enabled solely because the previous milestone is marked COMPLETED.

## First implementation tranche

Start with tests for repeated preprocessing, train-only PCA/scaling, split
consistency, and paired predictions; then build raw-count folds with manifests.
Preparing this infrastructure does not close R1/R6: real matrices, HPC execution,
new results, and human review remain outstanding. The operational registry
separates these states to avoid declaring a scientific milestone complete based
on code alone.
