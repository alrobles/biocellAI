# ADR-001 — Dataset for M1 (grounding mini-benchmark)

## Status
Proposed.

## Context
The core evaluation in the job description (and our RQ1) uses held-out
**donors**, so the M1 dataset needs ≥4 donors with reliable cell-type labels
and a manageable size (trainable locally on CPU or in a short HPC job).

## Options

| Option | Donors | Size | Pros | Cons |
|--------|--------|------|------|------|
| `pbmc3k` (scanpy built-in) | 1 | 2.7k | Frictionless smoke test | No multiple donors → unsuitable for real evaluation |
| Tabula Sapiens (cellxgene) | ~15 | millions → subset | Expert labels, multiple tissues | Download through cellxgene_census; filtering required |
| SEA-AD subset | ~80+ | 1.2M total | Closest alignment with the job description | Large; access through Allen SDK; M3+ |
| scIB Immune Human | ~10 | ~33k | Standard integration benchmark | Batch effects confound grounding |

## Decision

- **Plumbing/smoke test**: `pbmc3k` (one command, validates the end-to-end pipeline).
- **Actual M1 experiment**: **Tabula Sapiens blood** through `cellxgene_census`,
  filtered to ~20-50k cells, ≥4 donors, and cell types with ≥100 cells.
- SEA-AD is deferred to M5 if the method shows a signal (hybrid domain decision).

## Consequences

- Dependency on `cellxgene-census` (the `[data]` extra).
- Subset manifest in `data/manifests/` (query + filters + counts per donor),
  making the split auditable.
