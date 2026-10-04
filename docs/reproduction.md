# Independent reproduction — v2 protocol

How to reproduce the corrected (v2) results without access to our cluster.
Nothing here requires redistributed data: every artifact is either
generated from public sources or described by manifests + hashes.

## 0. Environment

```bash
git clone https://github.com/alrobles/biocellai-devel
cd biocellai-devel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data,text]"
```

CPU-only by design (all 34 confirmation jobs ran on CPU nodes).
Runtime: ~15 min per (cohort × seed) run; the full confirmation DAG is
~4 node-hours plus permutation arrays (~2 min per task, 900 tasks).

## 1. Source data (not redistributed)

| Cohort | Source | Access |
|---|---|---|
| Tabula Sapiens blood | CZ CELLxGENE Census | public, CC-BY-4.0 |
| SEA-AD (MTG, PFC) | Allen Brain Cell Atlas | public, **CC BY-NC 4.0** — non-commercial only |
| ROSMAP multiregion | AD Knowledge Portal (Synapse) | controlled — DUO application required |

`experiments/v2_revalidation/fold_manifests.json` and the per-cohort
`folds/fold_s*.json` manifests record exact source paths, SHA-256 of the
input matrices, code hashes, and the donor lists used — so a reproduction
can verify it is operating on byte-identical inputs.

## 2. Folds, captions, run matrix

Per cohort × seed, in order:

1. **Extract + fold** — `scripts/slurm/v2_confirm_prep.sbatch`
   (`v2_prepare.py` for the exploratory seeds): donor-held-out split
   (~25% test donors), 2000 train-only HVGs, per-cell normalization.
2. **Captions** — `scripts/slurm/v2_confirm_captions.sbatch`: T2 marker
   captions from train-only Wilcoxon; T3 copies the pre-registered
   PubMed corpus (`data/text/m7/`).
3. **Arm matrix** — `scripts/slurm/v2_confirm_run.sbatch`: all nine arms
   (B0,B1,B2,T0,T1,T2,T3,N1,N2), same cells/split/capacity/budget.
   Exports per-donor predictions + checkpoints.

## 3. Permutation nulls, transfer, inference

- `scripts/slurm/v2_permnull.sbatch` — one array task per permutation
  index (100/seed); shared donor→label maps
  (`perm_seed = seed*100003 + index`). B2 retrains on permuted
  endpoint labels (endpoint-supervision null); other arms permute the
  readout on frozen features.
- `scripts/v2_b2_readout_null.py` — revision check: readout-perm on the
  real B2 checkpoint (centers at ~0; verifies machinery).
- `scripts/slurm/v2_transfer.sbatch` /
  `v2_confirm_transfer_pfc.sbatch` — strict source-fitted transfer
  (encoder, PCA, ridge, axis all frozen on source).
- `scripts/slurm/v2_confirm_infer.sbatch` → `scripts/v2_inference.py` —
  paired donor-clustered bootstrap (2000), BH-FDR, gates →
  `inference/paired_comparisons.csv` + `inference/gates.json`.

## 4. Declarative driver (recommended)

The whole confirmation chain is encoded as a DAG:

```bash
python scripts/v2_pipeline.py --pipeline confirmation --dry-run   # review 34 jobs
python scripts/v2_pipeline.py --pipeline confirmation --submit    # SLURM
python scripts/v2_pipeline.py --pipeline confirmation --validate  # artifacts
```

Non-SLURM environments can run the same scripts directly; the sbatch
files are thin wrappers exporting variables.

## 5. Paper figures and tables

```bash
python scripts/v2_paper_figures.py   # paper/figures/*.pdf + tab_*.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every number in the manuscript is regenerated from committed artifacts —
no hand-copied values.

## What ships vs what does not

Per `spec/redistribution_manifest.json`: code, manifests, predictions,
nulls, captions, metrics and figures are released; source matrices,
ROSMAP/SEA-AD-derived weights and third-party weights are not. The
Hugging Face dataset `alrobles/biocellai-foundations` carries the
permitted artifact set.
