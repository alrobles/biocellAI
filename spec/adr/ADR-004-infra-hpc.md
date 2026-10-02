# ADR-004 — Infrastructure: local development + KU HPC

## Status
Proposed.

## Context
No local GPU (30GB RAM). KU HPC is available with tested PyTorch containers
(`pytorch_cuda_lite.sif`, `pytorch_rocm.sif` in `${SCRATCH_ROOT}/containers`),
Ollama on GPU nodes, and a documented Apptainer+SSH+tunnel pattern
(`knowledgebase/hpc-scripts/ARCHITECTURE.md`).

## Decision

- **Local**: development, unit tests, and a `pbmc3k` smoke test (CPU).
- **KU HPC**: all actual training from M1 if it takes >30 min locally.
  - `SCRATCH_ROOT=/beegfs/a474r867/biocellai` (new, same layout as
    `ancestral_embeddings`).
  - Cluster repository: `${SCRATCH_ROOT}/repos/biocellai-devel` (git pull + sbatch
    from the repository root, using the `SLURM_SUBMIT_DIR` pattern).
  - Existing containers; data in `${SCRATCH_ROOT}/data/`.
- **Local environment**: `pyproject.toml` with `[data,text,dev]` extras; HPC uses
  the container rather than the user's pip installation.
- Jobs through Hermes/hydra MCP or direct `sbatch`; logs → `results/m<N>/`.

## Consequences

- Committed scripts in `scripts/slurm/` (job→artifact traceability).
- No large datasets in git: `data/` contains only manifests, and `results/`
  contains only reports/metrics (no large embeddings).
