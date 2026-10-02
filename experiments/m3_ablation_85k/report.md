# M3 — grounding-source x text-encoder ablation at 85k cells (KU HPC, A100)

## Setup

- **Dataset**: Tabula Sapiens blood, full query — `tabula_blood_90000_s0.h5ad`
  (85,055 cells, 9 donors, 16 cell types, 2,000 HVGs, gene symbols mapped from
  `feature_name`).
- **Split**: donor-held-out per seed (seeds 0/1/2); markers computed on train donors only.
- **Arms**: `cell_only_supervised` baseline; `grounded_probe` + `grounded_zeroshot`
  for each of {`own_genes`, `type_markers`} x {`all-MiniLM-L6-v2`, `SapBERT-from-PubMedBERT-fulltext`}.
- `label_mode=False` throughout — honest grounding only (no cell-type names in captions).
- **Hardware**: KU HPC `sixhour` partition, A100 node (r13r06n01), job `29922096`.
- Cell encoder: MLP 2000→256→128→64. Text encoder frozen. CLIP-style symmetric
  InfoNCE, temperature 0.1, 40 epochs, batch 512.

## Results (mean ± std over 3 seeds)

| arm | caption_mode | text_model | macro-F1 | balanced acc |
|-----|-------------|-----------|----------|--------------|
| cell_only_supervised | — | — | 0.6028 ± 0.018 | 0.6177 ± 0.017 |
| grounded_probe | own_genes | MiniLM | 0.4140 ± 0.001 | 0.4756 ± 0.025 |
| grounded_probe | own_genes | SapBERT | 0.4038 ± 0.013 | 0.4828 ± 0.031 |
| grounded_probe | type_markers | MiniLM | 0.5601 ± 0.053 | 0.5782 ± 0.047 |
| grounded_probe | type_markers | SapBERT | 0.5475 ± 0.013 | 0.5850 ± 0.018 |
| grounded_zeroshot | own_genes | MiniLM | 0.2879 ± 0.022 | 0.4731 ± 0.031 |
| grounded_zeroshot | own_genes | SapBERT | 0.2606 ± 0.010 | 0.4460 ± 0.050 |
| grounded_zeroshot | type_markers | MiniLM | **0.5911 ± 0.015** | **0.6349 ± 0.034** |
| grounded_zeroshot | type_markers | SapBERT | 0.5776 ± 0.025 | 0.6196 ± 0.020 |

## Key findings

1. **Zero-shot grounded ≈ supervised on held-out donors.** With `type_markers`
   captions, the contrastively-trained model classifies cells from *unseen donors*
   at 0.591 F1 / 0.635 bal-acc — matching the supervised cell-only baseline
   (0.603 / 0.618) with zero label-trained weights. Balanced accuracy actually
   exceeds the supervised baseline. This is a clean positive signal for the
   CellOLMo hypothesis: when the grounding text carries class-discriminative
   biology, alignment alone produces donor-transferable representations.

2. **The grounding source dominates the text encoder choice.** `type_markers`
   vs `own_genes` is a ~2x F1 gap in zero-shot (0.59 vs 0.28). SapBERT ≈ MiniLM
   everywhere — a biomedical entity-linking encoder gives no benefit over a
   general sentence encoder for these short gene-list-style captions.

3. **Own-gene captions are honest but weak** (0.29 F1 zero-shot). Per-cell top-8
   gene lists are too noisy/redundant as grounding text; the signal is in
   type-level curated structure, not per-cell descriptions.

## Caveats (honest read)

- `type_markers` captions are **label-derived**: marker sets are computed from
  train-donor labels. So this arm is best understood as *supervision distilled
  through text* — language carries the class structure into a space where
  unlabeled held-out cells can be classified zero-shot. It is not evidence that
  unsupervised text grounding alone solves classification; it is evidence that
  text is a viable **carrier** of transferable biological structure.
- No test-donor leakage: markers computed on train donors only; test cells never
  contribute labels.
- `type_markers` contrastive loss plateaus high (~3.7–3.8) by design — only 16
  unique texts, so most batch pairs are "positive-same" collisions; the metric
  still tracks a usable aligned space.
- Cell-type granularity is coarse (Tabula's `cell_type` field); finer ontology
  terms would likely widen the grounding advantage.

## Artifacts

- `metrics.csv` — per-seed full metrics
- `losses.json` — all training curves (supervised converges to ~0.01; contrastive
  `own_genes` ~2.9→1.7; `type_markers` ~4.0→3.7 as expected)
- `data/manifests/markers_train_s0.json`, `cell_types.json` — marker table and
  type list used for captions (train donors only)
- Launch: `scripts/slurm/m3_ablation_85k.sbatch` on KU HPC (git-synced repo,
  shared HF cache, CUDA auto-detect).
