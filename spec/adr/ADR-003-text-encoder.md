# ADR-003 — Text encoder and caption source for M1

## Status
Proposed.

## Context
The language side of grounding involves two decisions: (a) how to generate
text for each cell, and (b) which encoder maps it into the shared space.

## Options — captions

| Option | Example | Pros | Cons |
|--------|---------|------|------|
| Template markers | "Blood cell expressing CD3D, CD8A, GZMK; cell type: CD8 T" | Deterministic, no LLM, auditable | "Trivial" text — risk of label leakage |
| Free-text cell type | Prose description of the cell type | Closer to CellWhisperer | Requires curation or an LLM |
| LLM-generated | Qwen/DeepSeek generates a description through Ollama | Scales to M4 | Nondeterministic, external dependency |

## Options — text encoder

| Option | Size | Pros | Cons |
|--------|------|------|------|
| `all-MiniLM-L6-v2` | 22M | Already used in `knowledgebase/rag`, fast | General-domain, not biomedical |
| SapBERT (UMLS) | ~110M | Biomedical entity embeddings | Heavier |
| PubMedBERT | ~110M | Pretrained on PubMed abstracts | Without fine-tuning, may underperform ST |
| Ollama embeddings (Qwen) | large | Already deployed on HPC | Latency, service dependency |

## Decision

- **M1**: captions = deterministic marker template; encoder = frozen sentence
  transformer (start with MiniLM, ablate SapBERT in M2).
- Two template variants deliberately **include and exclude** the cell-type name
  to measure how much of the effect is leakage versus actual grounding.
- LLM-generated captions + Ollama come in M4 (RQ3).

## Consequences

- `src/biocellai/captions.py` generates text from `adata.var`/`adata.obs` — pure and testable.
- Dependency on `sentence-transformers` (the `[text]` extra).
