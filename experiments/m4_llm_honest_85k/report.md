# M4 — LLM-generated biological descriptions as grounding text (85k cells)

## Setup

- **Generator**: `Qwen/Qwen2.5-14B-Instruct` (bf16), temperature 0.2,
  max 140 new tokens — `scripts/m4_generate.py` on KU HPC (l40 node,
  job `29922136`).
- **Prompt variant `honest`**: the model sees ONLY the train-derived marker
  list per type (never the type name). Output post-processed to strip any
  literal type-name occurrence; flagged in provenance.
- **Experiment**: `caption_mode=type_llm`, both text encoders, seeds 0/1/2,
  same 85k-cell dataset and donor-held-out protocol as M3.
- Zero-shot class captions remain marker-list captions (as M3) — so the eval
  is *not* a trivial same-string match between train and class text.

## Results (mean ± std, 3 seeds)

| arm | text_model | macro-F1 | balanced acc |
|-----|-----------|----------|--------------|
| cell_only_supervised | — | 0.6050 ± 0.030 | 0.6384 ± 0.013 |
| grounded_probe | MiniLM | 0.5310 ± 0.046 | 0.5531 ± 0.030 |
| grounded_probe | SapBERT | 0.5494 ± 0.029 | 0.5719 ± 0.022 |
| grounded_zeroshot | MiniLM | 0.3240 ± 0.049 | 0.4358 ± 0.059 |
| grounded_zeroshot | SapBERT | **0.4006 ± 0.044** | 0.4871 ± 0.015 |

Provenance: `data/text/type_desc_qwen_honest.{json,prov.json}` —
1/16 outputs contained the literal type name (platelet → stripped);
`prov.json` stores prompts + raw outputs + per-type leak flags.

## Reading

- Honest LLM prose (markers→prose, no label) **improves over per-cell
  `own_genes` captions** (zeroshot 0.40 vs 0.29 with SapBERT): the LLM
  correctly infers lineage/function from markers alone ("megakaryocyte
  developmental continuum", "myeloid origin", "antibody-secreting program")
  and that semantic structure grounds better than noisy per-cell gene lists.
- It does **not** reach `type_markers` zero-shot (0.59) — largely a
  *format-consistency* effect: zero-shot class captions are marker lists,
  so marker-list train captions transfer best. The honest LLM arm is the
  only one where train and eval text differ in register.
- SapBERT > MiniLM here (0.40 vs 0.32 zeroshot) — biomedical pretraining
  helps when captions are natural prose, unlike the gene-list captions in M3.

## Artifacts

- `metrics.csv`, `losses.json`, `data/text/type_desc_qwen_honest.*`
- See `experiments/m4_llm_labeled_85k/` for the leakage-probe counterpart.
