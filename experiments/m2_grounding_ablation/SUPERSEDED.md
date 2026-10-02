# SUPERSEDED — do not cite these numbers

This local M2 run completed before the gene-symbol cache-hit fix
(commit `28dd4b9`). Its captions were generated from a cached h5ad whose
`var_names` were positional indices (`"0"`, `"1"`, ...), so the "grounding"
text carried no gene semantics — the results in `metrics.csv`/`losses.json`
here are scientifically invalid.

The same ablation matrix was rerun correctly at larger scale on KU HPC:
see `experiments/m3_ablation_85k/` (85,055 cells, A100, job 29922096),
which supersedes this directory entirely.
