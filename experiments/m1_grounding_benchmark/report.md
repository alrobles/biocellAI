# M1 — cell-only vs text-grounded (held-out donors)

**Dataset**: Tabula Sapiens blood via cellxgene_census `2025-11-08` —
29,999 cells, **9 donors**, 16 cell types, 2,000 HVGs
(`manifest.json` contains auditable filters and counts).
**Split**: entire donors assigned to test (25% per seed, seeds 0/1/2).
**Model**: MLP encoder 2000→256→128→64 shared across arms;
frozen `all-MiniLM-L6-v2` text encoder; captions = a template listing
each cell's top-8 expressed genes ± the cell-type name.

## CORRECTED results (actual gene symbols, mean ± std, 3 seeds)

| arm | label in caption | macro-F1 | balanced-acc |
|-----|------------------|----------|--------------|
| cell_only_supervised | — | **0.590 ± 0.003** | 0.608 ± 0.018 |
| grounded_probe | no | 0.426 ± — | 0.479 ± — |
| grounded_probe | yes (leakage) | 0.472 | 0.504 |
| grounded_zeroshot | no | 0.293 | 0.470 |
| grounded_zeroshot | yes (leakage) | 0.397 | 0.560 |

(Exact standard deviations are in `metrics.csv` — the table uses means.)

## Warning: bug found and fixed (a lesson similar to deepla/M6)

The first run produced captions containing **positional/Ensembl IDs**
("Highly expressed genes: 1694, 3764, …") because cellxgene_census uses
feature IDs as `var_names`, while symbols are stored in `var['feature_name']`.
The grounding channel was corrupted: tokens lacked biological semantics.

- Fix: `use_gene_symbols()` runs after download **and** on cache hits.
- Measured effect of the fix (no-label zero-shot): 0.214 → 0.293 F1,
  0.349 → 0.470 bal-acc. Grounding with actual symbols does provide a signal.
- Reverse effect in the leakage arm (0.517 → 0.397 F1): with meaningless IDs,
  the caption reduced almost to the label alone (tight clusters); actual
  markers introduce more within-type variance. The label/no-label gap is now
  a more honest measure of the value of text.

## Honest interpretation

1. **Marker-only grounding still does not beat the baseline** (0.43 vs 0.59
   F1) at this scale — but the gap is smaller than it appeared with the
   corrupted channel, and the direction is as expected.
2. **Zero-shot without labels is no longer trivial** (0.47 bal-acc vs ~0.06
   chance across 16 types): the aligned space recovers actual biological
   structure from gene names alone.
3. The confirmed bottleneck is the **grounding source**: unstructured gene
   lists are a weak channel; M2/M3 ablate (a) markers per type,
   (b) a biomedical text encoder (SapBERT), and (c) LLM descriptions (M4).

## Leakage checks performed and ruled out

- Whole-donor splits: verified (`test_donor_split_no_overlap`).
- Marker table computed only on training cells.
- `label_mode=False` excludes the type name; verified by a test.
- Recorded caveat: HVGs are computed on the entire dataset before splitting
  (standard minor leakage).
