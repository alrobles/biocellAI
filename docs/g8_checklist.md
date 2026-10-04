# G8 — open release checklist (ADR-007)

Scope: release of data manifests, code, permitted weights (HF), and
paper tables under the redistribution rules in
`spec/redistribution_manifest.json`. Status: **checklist complete,
pending owner sign-off on the public tag**.

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | LICENSE adopted (Apache-2.0) | done | `LICENSE` |
| 2 | NOTICE with CC-BY-4.0 artifact clause + source terms | done | `NOTICE` |
| 3 | Per-artifact redistribution rules | done | `spec/redistribution_manifest.json` |
| 4 | Code + specs + tests public | done | `alrobles/biocellAI` |
| 5 | Dataset manifests + donor-split lists + hashes public | done | `fold_manifests.json`, `dataset_flow.csv` |
| 6 | Generated artifacts public (captions, metrics, predictions, nulls, gates) | done | HF `alrobles/biocellai-foundations`; v2 corpus staged + scan clean (1446 files, dry-run) |
| 7 | Permitted weights on HF with model cards | partial | only Tabula-trained weights may ship (Apache-2.0); SEA-AD weights need CC BY-NC decision; ROSMAP weights never ship |
| 8 | Paper tables/figures reproducible from artifacts | done | `scripts/v2_paper_figures.py`, `docs/reproduction.md` |
| 9 | Independent reproduction instructions | done | `docs/reproduction.md` |
| 10 | Zenodo DOI + CITATION.cff | done | `10.5281/zenodo.23103004`, `CITATION.cff` |
| 11 | No controlled-access data or secrets in release channels | done | whitelist + secret scan in `scripts/hf_release.py`; ROSMAP/SEA-AD matrices excluded |
| 12 | Public tag for the corrected-evidence release | pending | propose `v0.2.0` (v1.0 reserved for the peer-reviewed paper) |

## Release contents (corrected v2 evidence)

- Arm-matrix runs: `experiments/v2_revalidation/`, `experiments/v2_confirmation/`
- Inference: `paired_comparisons.csv`, `gates.json`, per-donor
  predictions, 900 permutation-null rows + 300 B2 readout-perm rows
- Transfer: strict source-fitted metrics, all directions/seeds
- Marker recovery: P@10 + coverage ceilings + manifests
- Paper: `paper/main.pdf` + generated figures/tables

## Explicitly NOT released

- ROSMAP/SEA-AD fold matrices and donor tables (controlled/CC BY-NC)
- Checkpoints trained on ROSMAP (never) or SEA-AD (pending NC decision)
- Third-party model weights (link-only)
- Raw source datasets
