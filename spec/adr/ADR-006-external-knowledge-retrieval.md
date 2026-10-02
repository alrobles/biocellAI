# ADR-006: External-knowledge grounding via PubMed literature retrieval

- Status: accepted
- Date: 2026 (M6)

## Context

M3/M4 decomposed the grounding benefit: `type_markers` ≈ supervised is
supervision distilled through text (markers derive from train labels);
label-free sources (`own_genes`, honest LLM prose) recover only 48–66% of
that effect. ADR-005 chose external-knowledge grounding as the next
direction. The question: can text that is *independent of our train labels*
carry the biological structure?

The lab asset that differentiates this line is the PubMed corpus at
`/beegfs/a474r867/litdump/pubmed/pubmed_full.db` — 30.8M articles with an
FTS5 index over (title, abstract, mesh), built by ecoseek-litdump.

## Decision

Retrieval-first grounding, measured before modeling:

1. **Query ladder per cell type** (`retrieval.marker_queries`):
   - L1 strict AND of top-3 markers (precision)
   - L2 pairwise ANDs of top-5 markers (co-occurrence)
   - L3 marker AND context (cell/cells/cellular)
   - L4 HGNC alias-expanded OR queries (recall boost — FTS5 has no
     stemming, so synonyms matter: GNLY ↔ granulysin)
   - L5 (name arms only) cell-type phrase + `mesh:` column-filtered
     queries via MESH_HINTS
2. **Pool by pmid, rerank** by marker coverage (dominant), immune-context
   prior (IMMUNE_CONTEXT vocabulary — label-free topical prior),
   aggregate reciprocal-rank, best BM25.
3. **Recovery-rate metrics first**: per type × arm we record n_pooled,
   n_returned, coverage distribution, caption length. Retrieval quality
   is a first-class result — a grounding failure caused by a retrieval
   failure must be distinguishable from a mechanism failure.
4. **Arms** (query side / output side):
   - `rag_markers`: marker queries, type name stripped from output —
     label-derived *query*, label-free *text*
   - `rag_markers_alias`: + HGNC expansion (the recall experiment)
   - `rag_name`: name-phrase + MeSH queries, raw output — upper bound
   - `rag_name_stripped`: name queries, name stripped from output —
     external text with weakest label signal
5. **Matched-register eval** (`--class-descriptions`): zero-shot class
   captions come from the same retrieved text — the M4 register-mismatch
   fix. Marker-register eval is still reported for M3 comparability.
6. Captions reuse the `type_llm` contract ({type: text}) — no new
   training machinery.

## Leakage accounting

| arm | query uses label | output contains label | verdict |
|-----|------------------|-----------------------|---------|
| rag_markers | markers (label-derived) | stripped | label-free text, label-derived selection |
| rag_markers_alias | markers + HGNC aliases | stripped | same |
| rag_curated | PanglaoDB markers (external) | stripped | **label-free end-to-end** |
| rag_curated_alias | PanglaoDB + HGNC aliases | stripped | **label-free end-to-end** |
| rag_name | type name + MeSH | raw | label-derived (upper bound) |
| rag_name_stripped | type name + MeSH | stripped | text is external; selection label-derived |
| m6_curated | — (PanglaoDB markers as captions directly) | markers | **label-free end-to-end** |

The fully label-free arms were added in M6 via PanglaoDB
(`scripts/build_panglao_map.py` → `data/manifests/panglao_markers.json`).

## Outcome (measured)

Pre-registered criterion **met**:

| arm | F1 | vs label-derived (0.591) | vs supervised (0.600) |
|-----|---:|---:|---:|
| rag_markers_alias | **0.563** | 95% | 94% |
| rag_curated_alias | **0.512** | 87% | 85% |
| m6_curated (markers direct) | 0.459 | 78% | 77% |
| rag_name (upper bound) | 0.506 | 86% | 84% |

Retrieval-recall metrics: HGNC aliases +72% pooled docs, +40%
high-coverage docs; curated external markers pool more docs than
train-derived markers. Marker-register eval of prose captions collapses
(0.07–0.20) — matched-register eval is required for literature text.
Caveat: all evals are 16 coarse blood types; the remaining question is
whether the advantage survives fine-grained/continuous phenotypes.
