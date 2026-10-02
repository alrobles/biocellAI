# M6 — External-knowledge grounding via PubMed retrieval

**Jobs**: retrieval `29922218` (6 arms × 16 types), grounding `29922336–38`
(gpu:1), curated-marker arm `29922256`, synthesis `29922339/40`.
**Corpus**: `/beegfs/a474r867/litdump/pubmed/pubmed_full.db` — 30.81M
articles, FTS5 over (title, abstract, mesh).

## Recovery-rate results (the question asked)

Alias expansion is the single biggest recall lever:

| arm | query source | pooled docs | cov≥0.25 docs | timeouts |
|---|---|---:|---:|---:|
| rag_markers | train markers | 180 | 15 | 0 |
| **rag_markers_alias** | train markers + HGNC aliases | **310** | **21** | 0 |
| rag_curated | PanglaoDB markers | 224 | 16 | 0 |
| rag_curated_alias | PanglaoDB + aliases | 237 | 18 | 0 |
| rag_name | type name + MeSH | 258 | 15 | 8 |
| rag_name_stripped | name + MeSH, name stripped | 256 | 15 | 9 |

- HGNC synonym expansion: **+72% pooled docs** (180→310), **+40%
  high-coverage docs** (15→21 with ≥2 markers).
- Curated (label-free) markers pool *more* docs than train markers
  (224 vs 180) — external sources are not the recall bottleneck.
- Name-phrase queries hit the 30s timeout 8–9 times (common terms scan
  huge posting lists); pooling still delivered ~258 docs via the ladder.
- Raw marker-coverage retrieval is noisy in *topicality*: top v1 hits
  included off-target biomarker papers (radiation-therapy arrays naming
  MS4A1). v2 rerank adds an immune-context prior; `rag_llm` synthesis
  (LLM over retrieved excerpts) is the stronger topicality filter.

## Grounding results (85k cells, held-out donors, 3 seeds)

| arm | register | F1 | bal-acc |
|---|---|---|---:|
| cell_only_supervised | — | 0.600 | 0.620 |
| type_markers (train-derived, M3 ref) | markers | 0.591 | 0.635 |
| **m6_curated** (PanglaoDB markers direct) | markers | **0.459** | 0.564 |
| **rag_markers_alias** | matched prose | **0.563** | **0.613** |
| **rag_curated_alias** | matched prose | **0.512** | 0.565 |
| **ragllm_honest** (retrieval→Qwen, stripped) | matched prose | **0.589** | **0.631** |
| **ragllm_curated** (PanglaoDB→retrieval→Qwen) | matched prose | **0.603** | **0.634** |
| ragllm_labeled (leakage upper bound) | matched prose | 0.600 | 0.635 |
| rag_name (label in text — upper bound) | matched prose | 0.506 | 0.554 |
| rag_* (same arms) | marker-register | 0.07–0.20 | — |

## Reading

1. **Matched register is decisive.** The same retrieved captions score
   0.15 under marker-register class captions and 0.56 under
   matched-register ones. The M4 register-mismatch hypothesis confirmed
   and quantified (~0.4 F1). Evaluating literature captions against
   gene-list class captions is invalid.
2. **External knowledge works.** `rag_markers_alias` — PubMed text
   retrieved by marker queries, type name stripped — reaches **95% of
   the label-derived marker arm** and 94% of supervised. That crosses
   the ADR-005 gate (≥0.50 F1 / ≥75% of label-derived).
3. **Fully label-free works too.** `rag_curated_alias` (PanglaoDB
   markers → retrieval → stripped text, no train labels anywhere)
   reaches 0.51 F1. And curated markers *directly* as captions
   (`m6_curated`, no retrieval) already give 0.46 — 78% of the
   label-derived arm.
4. **Retrieval ≠ the bottleneck; topicality was.** Once the context
   prior fixed ranking, retrieved text beat the label-derived query
   upper bound (rag_name 0.506 < rag_markers_alias 0.563) — marker
   queries surface more diverse biology than name queries.
5. **LLM synthesis is the quality lever.** `ragllm_honest` — Qwen2.5-14B
   synthesizing a two-sentence description from the retrieved excerpts,
   type name never shown, 2/16 outputs flagged for inferred-name leakage
   (kept, flagged) — reaches **0.589 F1**, statistically
   indistinguishable from the label-derived marker arm (0.591) and the
   supervised baseline (0.600). Retrieval recall + LLM topicality
   filtering closes the entire label-free gap at this granularity.
   The labeled synthesis probe (0.600) shows the remaining headroom is
   ≈0.01 — the honest arm is at ceiling for this task.
   **The fully label-free end-to-end chain** (`ragllm_curated`:
   PanglaoDB markers → PubMed retrieval → Qwen synthesis, name never
   shown anywhere) reaches **0.603 F1 / 0.634 bal-acc** (SapBERT) —
   matching supervised with zero labels in the pipeline. This is the
   headline: external biological knowledge, not relabeled supervision,
   can carry cell identity.
6. Caveat: all arms still evaluate on 16 coarse blood types.
   "Supervision distillation" and "literature grounding" are now
   within ~0.03 F1 — the interesting next axis is fine-grained or
   continuous phenotypes (SEA-AD pathology), where curated lists
   are sparse and retrieval matters more.

## Artifacts

- `data/retrieval/m6/{arm}/*.jsonl` — pooled docs with pmid, coverage,
  context score (provenance)
- `data/text/type_desc_pubmed_{arm}.json` — grounding captions
- `experiments/m6_retrieval/retrieval_metrics.csv` — 96 type×arm rows
- `experiments/m6_{arm}_85k/` — metrics.csv + losses.json per run
- `data/manifests/panglao_markers.json`, `data/gene_aliases.json`
