# ADR-005: Post-M5 direction — external-knowledge grounding

## Status

Accepted (2026-09-20) — decision document for the next experimental phase,
informed by M1–M4 results (see `experiments/m5_consolidated/report.md`).

## Context

M3/M4 decomposed *why* caption grounding helps:

| source | label-free? | zeroshot F1 | what it tests |
|--------|-------------|-------------|----------------|
| own_genes | yes | 0.29 | per-cell description |
| type_markers | **no** (label-derived) | 0.59 | supervision via text |
| type_llm honest | yes | 0.40 | knowledge inferred from markers |
| type_llm labeled | no (name leaked) | 0.58 | leakage upper bound |

The mechanism is proven: text alignment transfers class structure to
held-out donors. The open gap: **no label-free source reaches supervised
performance**. That gap IS the research question worth pursuing.

## Options considered

### A. External-knowledge grounding (CHOSEN)

Replace label-derived captions with sources independent of train labels:

- **Cell Ontology / Uberon definitions** — canonical textual descriptions
  per cell type, no dataset labels involved.
- **Curated marker databases** — PanglaoDB, CellMarker: literature-derived
  marker sets, not computed from the train split.
- **Literature retrieval (RAG)** — the lab's unique asset: genominer's
  ~36M-abstract PubMed index + FAISS/sentence-transformer retrieval server
  (knowledgebase/rag). Grounding text *retrieved* from literature, not
  computed from labels.

Also required by our own finding: **matched-register eval** — class
captions must come from the same source family as train captions (e.g.,
ontology definitions for both), otherwise format mismatch caps the score
(M4 showed this costs ~0.2 F1).

Decision criterion (falsification gate): a label-free external source
reaching ≥0.50 zeroshot F1 on held-out donors would be a real result —
75%+ of the label-derived arm without touching labels. Below that, the
honest conclusion is that current grounding = distillation, publishable
as a critique-style finding.

### B. SEA-AD showcase (deferred)

Porting to the Alzheimer's atlas (donor×region splits, neuronal subtypes,
pathology axis) is the Ai2-aligned demonstration. Deferred: running the
same label-derived experiment on a harder dataset adds risk, not
information. Revisit after (A) establishes whether label-free grounding
can work at all.

### C. Eco-evo cross-species branch (deferred, high ceiling)

Grounding cell types across species where ontologies are sparse — the
setting where text grounding has the most to give and where this portfolio
would be genuinely novel (phylo + genomics assets exist in the lab).
Highest scientific ceiling, hardest execution; needs (A) first.

### D. Scale up model/data (rejected for now)

A bigger encoder or more cells would not answer the open question — the
bottleneck is grounding-source information content, not capacity.

## Consequences

- Next milestone M6: `external_kb` caption mode + matched-register eval.
- genominer/RAG integration becomes the differentiating infrastructure.
- The paper draft (`paper/`) frames the current state honestly: mechanism
  proven, label-free gap open — any positive external-KB result lands
  directly as the paper's central claim.

## Risks

- Ontology definitions may be too generic (immunology prose > marker-level
  detail) → mitigated by mixing sources (ontology + DB markers).
- Retrieval quality from PubMed is noisy → start with curated DBs, use RAG
  as augmentation not sole source.
- Cell types in Tabula are coarse — external knowledge may describe
  finer/coarser granularity → document granularity mismatches explicitly.
