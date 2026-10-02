# BioCellAI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23103004.svg)](https://doi.org/10.5281/zenodo.23103004)

**Foundations** — a research toolkit for controlled cell–text representation learning.

BioCellAI investigates whether aligning single-cell expression profiles with
biological text improves generalization to unseen donors beyond equivalent
non-text supervision — and how much of any improvement survives controls for
cell composition, region, and cohort.

The core comparison is deliberately strict: same cells, same labels, same
encoder, same training budget — biological text versus non-text supervision
of equivalent strength.

This is an independent implementation of the research questions described for
**CellOLMo** (Ai2 × Allen Institute): open language models × single-cell brain
data on SEA-AD, evaluated on held-out donors across cell, region, and donor
levels. See [docs/cellolmo-alignment.md](docs/cellolmo-alignment.md) for the
artifact-level mapping to that agenda. Not affiliated with Ai2 or the Allen
Institute.

## Status

**Research alpha.** The data-preparation and evaluation infrastructure is
implemented and tested; the corrected v2 benchmark is still being executed.
The public repository is `alrobles/biocellAI` (tagged `v0.1.0-alpha` as an
infrastructure snapshot) and the Python package is `biocellai`;
**Foundations** is the codename for this infrastructure stage, not a
publication claim.

| Capability | State |
|---|---|
| Donor-level splits, train-only HVG selection, counts validation | Implemented; tested on synthetic data |
| Immutable fold builder with manifests, hashes, donor/gene lists | Implemented |
| Train-only PCA, feature scaling, ridge readouts | Implemented |
| Paired bootstrap and exact gate utilities | Implemented |
| One-hot / random-prototype supervision controls | Implemented as components |
| Integrated v2 runner (all control arms, equivalent budgets) | Pending |
| Corrected real-cohort benchmarks (blood, SEA-AD, ROSMAP) | Pending |
| Strict source-fitted cross-cohort transfer | Pending |
| Blinded biological caption review | Pending |

Full task registry and dependencies:
[`spec/revalidation_tasks.json`](spec/revalidation_tasks.json) —
`python scripts/revalidation_status.py` prints the live state.

## Installation

Requires Python ≥ 3.11. From source:

```bash
git clone https://github.com/alrobles/biocellAI
cd biocellAI
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,data,text]"
pytest -q   # 74 tests
```

Extras: `data` (scanpy/anndata/cellxgene-census), `text`
(sentence-transformers; downloads encoder weights on first use), `dev`.

## Quickstart: v2 fold preparation

`scripts/v2_prepare.py` builds immutable donor-held-out folds from a verified
count matrix. With any AnnData `.h5ad` whose `X` (or a named layer) contains
raw counts and `obs["donor_id"]` identifies donors:

```bash
python scripts/v2_prepare.py \
    --input cohort.h5ad --counts-layer X \
    --cohort seaad --source-release "SEA-AD MTG 2024" \
    --seeds 0 1 2 --out experiments/v2_revalidation/seaad_run1
```

Each run writes `fold_s*.h5ad` plus per-fold JSON manifests (source hash,
environment versions, code hashes, donor/gene lists, matrix hash) and refuses
to overwrite an existing `--out` directory. **Prepared ≠ evaluated**: the
`PREPARED.json` marker records `scientific_results_complete: false`.

Synthetic smoke test without real data:

```bash
python - <<'PY'
import numpy as np, pandas as pd
from anndata import AnnData
rng = np.random.default_rng(0)
AnnData(
    X=rng.poisson(3.0, (80, 400)).astype(np.float32),
    obs=pd.DataFrame({"donor_id": np.repeat([f"d{i}" for i in range(8)], 10)}),
    var=pd.DataFrame(index=[f"g{i}" for i in range(400)]),
).write_h5ad("/tmp/synth.h5ad")
PY
python scripts/v2_prepare.py --input /tmp/synth.h5ad --counts-layer X \
    --cohort synthetic --source-release local --seeds 0 --out /tmp/v2_folds
```

A historical end-to-end pipeline check exists (`python -m biocellai.experiment m1
--dataset pbmc3k`), but its PBMC donor IDs are randomly assigned pseudo-donors:
it exercises code paths only and demonstrates nothing about generalization.

## Input requirements

- **Counts**: `X` or a named layer must contain nonnegative, integer-valued
  counts. Numerical checks do not establish provenance — you must identify
  which layer holds verified raw counts. Preprocessed inputs (existing
  `log1p` state) are rejected.
- **Donors**: `obs["donor_id"]` is required; splits are donor-level and
  unified across seeds (`split_donor_ids`).
- **Genes**: unique identifiers; symbol mapping via `data/gene_aliases.json`.
- HVGs are selected on **train donors only**; normalization happens before
  the gene subset; the fold records `biocellai_data_state` and protocol metadata
  in `uns`.

## Workflow

```
verified counts → v2_prepare (frozen folds + manifests)
               → v2 runner (text arms + equivalent-supervision controls)  [pending]
               → inductive readouts (train-fit PCA/scaler/ridge)
               → paired comparisons, permutations, exact gates
```

All scientific outputs go under `experiments/v2_revalidation/<run-id>/`.
Historical `experiments/m*` directories are preserved read-only in spirit:
the tooling refuses to overwrite prepared outputs, and historical runners
now reject renormalizing processed files.

## Scientific scope and limitations

- Grounding where a caption is assigned to a cell **by its label** is
  *text-mediated supervision*, not label-free learning — even when the text
  is external knowledge or omits the class name. The interesting question is
  what biological text adds **beyond** equivalent supervision; that is what
  the pending control arms measure.
- Historical M0–M12 results are exploratory. The audit found repeated
  preprocessing, transductive components, and non-equivalent comparisons;
  see [docs/milestones.md](docs/milestones.md) for the full log and caveats.
- Nothing here is a claim that text helps, that composition explains the
  pathology axis, or that encoder scale is irrelevant — those are open
  questions the v2 matrix is designed to answer.
- Cross-sectional postmortem pseudoprogression scores are not longitudinal
  trajectories.

## Data and artifacts

- Included: dataset manifests, gene aliases, retrieval corpora
  (`data/retrieval/`), captions/provenance (`data/text/`), specs, ADRs, code,
  tests, and historical milestone reports.
- **Not** included: raw single-cell datasets (Tabula Sapiens, SEA-AD,
  ROSMAP), model checkpoints, or large embeddings — these live on KU HPC and
  are subject to their own access terms. No large data or weights are stored
  in git.
- The `m1` pipeline check downloads `pbmc3k` and a text encoder on first run.
- Released artifacts: captions, retrieval corpora, manifests, and docs are on
  Hugging Face at
  [`alrobles/biocellai-foundations`](https://huggingface.co/datasets/alrobles/biocellai-foundations)
  (uploaded via `scripts/hf_release.py` — whitelist packaging + secret scan).

## Documentation

- [`spec/00-spec.md`](spec/00-spec.md) — project specification
- [`spec/adr/`](spec/adr/) — architecture decision records
- [`spec/adr/ADR-011-scientific-revalidation.md`](spec/adr/ADR-011-scientific-revalidation.md) — v2 protocol
- [`spec/revalidation_tasks.json`](spec/revalidation_tasks.json) — task registry (`scripts/revalidation_status.py`)
- [`docs/milestones.md`](docs/milestones.md) — historical M0–M13 log
- [`docs/cellolmo-alignment.md`](docs/cellolmo-alignment.md) — mapping to the CellOLMo research agenda
- [`docs/historical-results.md`](docs/historical-results.md) — honest review of exploratory results
- [`docs/release-plan.md`](docs/release-plan.md) — weights/data release plan and license stack
- [`paper/`](paper/) — manuscript (provisional; full rewrite pending v2 evidence)
- [`scripts/slurm/`](scripts/slurm/) — version-controlled KU HPC jobs

## License and citation

Code and released model weights are **Apache-2.0** (see `LICENSE`);
project-generated captions, retrieval corpora, and manifests are
**CC-BY-4.0** (see `NOTICE`). Third-party datasets keep their own terms —
controlled-access data are not redistributed. Per-artifact release
permissions are tracked in `R7-LICENSE` and
[docs/release-plan.md](docs/release-plan.md). If you use this repository in
work leading to a publication, cite it via the archived DOI
([10.5281/zenodo.23103004](https://doi.org/10.5281/zenodo.23103004)) and note
the Foundations codename.

## Credits / context

Developed within the alrobles ecosystem: KU HPC, Apptainer, Ollama, PubMed
indexing (genominer), and spec-driven development under `spec/`.
