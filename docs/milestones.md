# Historical milestone log (M0–M12 + composition diagnostic)

This file records the development history of the project. It is a lab
notebook, **not** a statement of validated results. All entries below predate
the v2 revalidation protocol ([ADR-011](../spec/adr/ADR-011-scientific-revalidation.md))
and must be read as exploratory evidence only.

| Milestone | Status | Artifact |
|-----------|--------|----------|
| M0 spec + scaffold | done | `spec/00-spec.md` + ADRs |
| M1 mini-benchmark (30k cells, local) | done | `experiments/m1_grounding_benchmark/` |
| M2 grounding-source × text-encoder ablation | superseded→M3 | `experiments/m2_grounding_ablation/SUPERSEDED.md` |
| M3 scale-up to 85k cells on KU HPC (A100) | done | `experiments/m3_ablation_85k/` |
| M4 LLM captions (Qwen2.5-14B, honest/labeled) | done | `experiments/m4_llm_*_85k/`, `data/text/` |
| M5 consolidation + go/no-go | done | `experiments/m5_consolidated/report.md` |
| M6 external-knowledge grounding (PubMed RAG) | done | `src/biocellai/retrieval.py`, `experiments/m6_retrieval/report.md` |
| M7 SEA-AD brain + progression (240k nuclei, 127 donors) | done (first pass) | `experiments/m7_seaad/report.md` |
| M8 donor-pathology retraining + target sweep (8 arms + nulls) | executed — gates under review | `experiments/m8_sweep/report.md` |
| M9–M11 SEA-AD pathology heads, ROSMAP replication, MIL/ordinal objectives, PubMed pathology captions | done | `experiments/m9_*`–`m11_*` |
| M12 scGPT frozen features + end-to-end fine-tuning | done | `experiments/m12_scfm/report.md` |
| M13 diagnostic: donor composition baseline | done | `experiments/m13_composition/` |

## Historical headline figures (exploratory)

The table above records historical execution; it does not certify gates under
the corrected protocol. Blood experiments reached F1≈0.59–0.60 through
**text-mediated supervision**: even when the content is external or omits the
class name, assigning it to cells uses training labels. In SEA-AD, C1 recorded
rho≈0.604–0.607; M12 did not exceed it in the configurations tested.
Composition alone recorded rho≈0.504, without establishing a causal
composition/state decomposition. These figures require revalidation with
consistent data, splits, transformations, and baselines; they are not v2 results.

## Methodological issues identified by the audit

- Repeated preprocessing of already-normalized `.h5ad` inputs.
- HVGs/PCA fit with test donors contributing (transductive components).
- Label supervision described as "label-free".
- Captions derived from seed-0 markers reused across other splits.
- Non-equivalent probe architectures and donor sets across arms.
- Geneformer V2 inputs not matching the official tokenizer protocol.
- Composition baseline missing from earlier headline comparisons.

The corrected protocol and its dependency graph live in
[`spec/revalidation_tasks.json`](../spec/revalidation_tasks.json).
