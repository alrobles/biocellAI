# Alignment with the CellOLMo research program

This repository is an **independent** implementation of the research questions
publicly described for CellOLMo (Ai2 × Allen Institute): whether grounding
single-cell models in biological language produces measurable improvements
over molecular-only or equivalent non-text supervision, evaluated on held-out
donors. It is not affiliated with Ai2 or the Allen Institute.

The table maps each element of the published research agenda to concrete
artifacts here. **Historical** = exploratory results exist but predate the v2
protocol. **Pending** = designed and tracked in the revalidation registry.

## Research challenges

| CellOLMo challenge | Evidence in this repository | State |
|---|---|---|
| Multimodal model combining gene expression with text, using an open LM | Contrastive cell encoder trained against frozen text encoders (`src/biocellai/model.py`, `src/biocellai/train.py`); OLMo-generated captions with full provenance (`data/text/*olmo*.prov.json`); Qwen2.5-14B caption sets | Historical |
| Reproduce and evaluate scGPT, Geneformer, CellWhisperer, C2S-Scale | `experiments/m12_scfm/` (scGPT frozen + fine-tuned), `scripts/g2_geneformer.py`, `src/biocellai/scgpt_enc.py`; audit tasks `R4-GENEFORMER`, `R4-SCGPT-CW`, `R4-C2S` | Historical + pending audit |
| Adapt approaches to brain / neurodegeneration data | SEA-AD pipeline (`scripts/build_seaad.py`, `experiments/m7_seaad/`, `m8_*`, `m9_*`); ROSMAP cross-cohort arm (`experiments/m11*`) | Historical |
| Controlled experiments: language grounding vs cell-only / text-free baselines | v2 arm matrix with equivalent budgets — expression-only (B2), text grounding (T0–T3), nulls (N1), one-hot/random prototypes (`src/biocellai/train.py`, `spec/revalidation_tasks.json` `R3-*`) | Pending (components implemented) |
| Region- and donor-level representations with pathology and progression | Donor MIL pooling (`scripts/m11_mil_pooling.py`), pseudoprogression/CPS, Braak, CERAD, Thal, ADNC targets (`experiments/m8_sweep_*`, `m9_*`), composition baseline (`experiments/m13_composition/`) | Historical |
| Held-out-donor evaluation | Unified donor splits (`split_donor_ids`), train-only HVG/PCA/scaler, inductive readouts (`src/biocellai/data.py`, `src/biocellai/progression.py`) | Implemented |
| Marker recovery | Independent marker-recovery task `R5-MARKERS` (precision@10, non-circular provenance) | Pending |
| Regional vulnerability | `experiments/m8_vulnerability_*` arms; type×region analysis planned in `R5-STATE` | Historical + pending |
| Expert review of plain-language outputs | Blinded two-reviewer rubric with adjudication, task `R5-REVIEW` | Blocked on reviewer assignment |
| Open release of code, data, weights | Spec-first development (`spec/`, ADRs), manifests + SHA-256 provenance, immutable outputs, release task `R8-RELEASE` | Partially implemented |

## Skills evidence for an application

- **Transformer training/adaptation**: contrastive multi-head training loop,
  frozen HF/SentenceTransformer encoders, scGPT fine-tuning path.
- **Python/PyTorch, reproducible code**: `src/biocellai/` package, 63-test suite,
  `pyproject.toml` extras, immutable fold builder with hashed manifests.
- **Single-cell tooling**: scanpy/AnnData pipeline, cellxgene-census queries,
  HVG selection, gene-symbol aliasing, donor-level QC.
- **Neurodegeneration domain**: SEA-AD + ROSMAP, neuropathology targets
  (CPS, Braak, CERAD, Thal, ADNC), regional composition controls.
- **Experimental rigor**: the project self-audited and replaced an optimistic
  protocol with a leakage-controlled v2 design — donor-isolated feature
  selection, equivalent-supervision controls, paired inference, and explicit
  separation of exploratory vs confirmatory evidence
  ([ADR-011](../spec/adr/ADR-011-scientific-revalidation.md)).

## Honest caveats for reviewers

- This is a **research alpha**: the controlled v2 benchmark is still running;
  no validated claim that language grounding helps is made here.
- Historical figures (F1≈0.59–0.60 blood; CPS ρ≈0.60; composition ρ≈0.50) are
  exploratory and labeled as such in [docs/milestones.md](milestones.md).
- Model weights and raw datasets are not in git; redistribution terms are
  pending (`R7-LICENSE`).
