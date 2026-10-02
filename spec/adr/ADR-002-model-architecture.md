# ADR-002 — Model architecture for M1

## Status
Proposed.

## Context
We need two comparable variants: cell-only and text-grounded, with grounding
as the **only** difference (same capacity, same budget). The job description
mentions scGPT/Geneformer (transformers over gene tokens), but for M1 the
priority is controlled experimental design, not a large architecture.

## Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| MLP on HVGs | expression vector → MLP → embedding | Minimal, fast, CPU-friendly | Less "LM-like" |
| Mini-Geneformer | rank-value tokens → small transformer | Aligned with the job's baseline | More code, slower, likely the same outcome at this scale |
| Pretrained scGPT | public checkpoint | Actual reference model | Heavy; distracts from the research question |

## Decision

- **M1**: shared MLP encoder for both variants (256→128→64).
  - *cell-only*: supervised cross-entropy on cell type.
  - *grounded*: CLIP-style contrastive cell↔caption alignment + the same classifier.
- Comparison with full Geneformer/scGPT models comes in M3, where the scale
  justifies HPC use.

## Consequences

- The key contract is `cell → embedding` and `caption → embedding` in a shared
  space; the architecture can grow without changing the experiment.
- Define it in `src/biocellai/model.py` with a configuration dataclass (hidden dims, dropout).
