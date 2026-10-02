# biocellai-devel — Master specification

> Experimental sandbox at the intersection of open language models and biology.
> External reference: Ai2 "Young Investigator, Open Language Models for Biology"
> (CellOLMo) — Greenhouse job 8090222.

## 0. Context

CellOLMo (Ai2 × Allen Institute) investigates whether an open LM (OLMo) *grounded*
in biological language and knowledge measurably improves on models trained only
on molecular data, applied to the SEA-AD single-cell atlas (Alzheimer's disease).
Reference baselines: scGPT, Geneformer, CellWhisperer, C2S-Scale. Evaluation:
held-out donors, cell-type/marker recovery, regional vulnerability, and
expert-reviewed natural-language QA. Everything open (code + data + weights).

## 1. Objective of biocellai-devel

An **aligned but differentiated** experimental portfolio: demonstrate the ability
to formulate and investigate CellOLMo's central question at a manageable scale,
with honest experimental rigor (including negative results), drawing on existing
components from the alrobles ecosystem:

- FTS5 index of ~36M PubMed abstracts (`genominer`) — a biological knowledge
  corpus for grounding.
- pLM embedding extraction (ESM-2) on KU HPC (`ancetralembbedings`).
- Measured LoRA/post-training pipeline (`nemotron-eco-reasoner`).
- FAISS + sentence-transformers RAG server (`knowledgebase/rag`).
- Ollama on GPU nodes (Qwen-35B, DeepSeek) using the Apptainer+tunnel pattern.

**Non-goal**: compete with Ai2's scale or solve CellOLMo. The value lies in
experimental quality, not model size.

## 2. Research questions

- **RQ1** (core, mirroring CellOLMo): does a cell representation aligned with
  biological text improve on a cell-only baseline in downstream tasks
  (cell-type classification) evaluated on **held-out donors**?
- **RQ2**: which grounding source contributes most — marker-gene template
  captions, free-text cell-type descriptions, or structured knowledge (GO/marker DB)?
- **RQ3** (differentiator, later): does grounding in aggregated knowledge
  (literature via RAG / phylogenetic context) capture information that local
  captions do not?

## 3. Milestones and acceptance criteria

| M | Scope | Acceptance criterion |
|---|-------|----------------------|
| M0 | Spec + scaffold | This specification + ADRs approved; `pip install -e .` + `pytest` pass |
| M1 | Grounding mini-benchmark | Reproducible script training cell-only vs text-grounded models on a small multi-donor dataset; `experiments/m1_*/report.md` with metrics + honest limitations |
| M2 | Grounding sources | A/B/C comparison of text sources (template markers / cell-type free text / structured GO); report with an ablation table |
| M3 | HPC scale-up | The same experiment running on KU HPC (Slurm + Apptainer) on a ~100k-cell dataset; job→artifact traceability |
| M4 | RAG-grounded variant | Captions enriched through PubMed-index retrieval (or GO) vs local captions; evaluation of RQ3 |
| M5 | Consolidation | Honest results document in the style of `deepla/m5`; go/no-go decision on SEA-AD or an eco-evo direction |

## 4. M1 design (mini-benchmark)

```
multi-donor dataset (Tabula Sapiens blood or cellxgene subset ~20-50k cells)
        │
        ├── Cell-only baseline: MLP encoder on normalized HVGs
        │      → embedding → cell-type classifier (supervised)
        │
        └── Grounded: CLIP-style contrastive learning
               cell encoder + text encoder (frozen, biomed: SapBERT/PubMedBERT)
               captions = deterministic template (top markers + cell type + tissue)
               → aligned embedding → zero-shot / few-shot cell-type classification
        │
        ▼
   Eval: split by donor_id (held-out donors)
   metrics: macro-F1, balanced acc, marker-gene recovery
   seeds: ≥3
```

Open decisions → `spec/adr/ADR-001..004`.

## 5. Verification and discipline

- Each milestone produces a `report.md` with measured, not aspirational, numbers.
- Negative results are reported as such (lesson from deepla/M6: ground-truth
  leakage inflated metrics — actively look for leakage).
- Split by donor before any tuning; never test on training donors.
- Seeds are fixed and recorded in the report.
- HPC jobs have a committed `.sbatch` script + log → artifact in `results/`.

## 6. Risks

| Risk | Mitigation |
|------|------------|
| Small dataset with insufficient donor structure | Select a dataset with ≥4 donors in ADR-001; verify before training |
| Trivial captions = text encoder learns the label | That *is* the question: report whether trivial grounding already wins (an honest result) |
| Insufficient local compute | Move all training to KU HPC from M1 if the local MLP takes >30 min |
| Premature over-engineering | M1 = minimum viable encoder; full transformer models come in M3+ |
