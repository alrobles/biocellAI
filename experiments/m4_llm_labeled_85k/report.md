# M4-labeled — leakage probe: LLM descriptions that name the cell type

**Not an honest grounding result.** The generator was told each cell type's
name (`LABELED_PROMPT`), so the prose contains it — a soft analogue of
`include_label=True` from M1, but with natural text instead of a template.

- Generator: `Qwen/Qwen2.5-14B-Instruct`, temp 0.2 (job `29922137`,
  pro6000 node). Provenance: `data/text/type_desc_qwen_labeled.prov.json`.
- Same 85k dataset / donor split / seeds as M3.

## Results (mean ± std, 3 seeds)

| arm | text_model | macro-F1 | balanced acc |
|-----|-----------|----------|--------------|
| cell_only_supervised | — | 0.6028 ± 0.012 | 0.6162 ± 0.014 |
| grounded_probe | MiniLM | 0.5101 ± 0.033 | 0.5299 ± 0.047 |
| grounded_probe | SapBERT | 0.5645 ± 0.036 | 0.5794 ± 0.029 |
| grounded_zeroshot | MiniLM | 0.5745 ± 0.040 | 0.6251 ± 0.023 |
| grounded_zeroshot | SapBERT | 0.5533 ± 0.045 | 0.6011 ± 0.021 |

## Reading

- The name-in-prose leak does **not** beat `type_markers` zero-shot
  (0.575 vs 0.591 MiniLM): even with the label leaked, register mismatch
  between train captions (prose) and class captions (marker lists) caps
  transfer. Format consistency between train/eval text matters as much as
  whether the label literally appears.
- Useful upper-bound reference: a *matched-format* labeled arm would need
  prose class captions too — left as future work since the honest arms are
  the scientifically meaningful ones.
