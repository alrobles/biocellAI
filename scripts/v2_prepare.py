from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import anndata as ad

from biocellai import data
from biocellai.data import (
    prepare_donor_fold,
    split_donor_ids,
    use_gene_symbols,
    validate_counts,
)
from biocellai.revalidation import file_sha256, ordered_ids_sha256, reserve_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--counts-layer", required=True, help="X or the verified raw-count layer name")
    p.add_argument("--cohort", required=True)
    p.add_argument("--source-release", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seeds")
    out = reserve_output(args.out)
    a = ad.read_h5ad(args.input)
    if args.counts_layer != "X":
        a = ad.AnnData(X=a.layers[args.counts_layer].copy(),
                       obs=a.obs.copy(), var=a.var.copy())
    validate_counts(a)
    a = use_gene_symbols(a)
    source = {
        "path": str(args.input.resolve()), "sha256": file_sha256(args.input),
        "counts_layer": args.counts_layer, "cohort": args.cohort,
        "release": args.source_release,
        "n_cells_input": a.n_obs, "n_genes_input": a.n_vars,
        "cells_per_donor_input": a.obs["donor_id"].astype(str).value_counts().to_dict(),
    }
    environment = {name: version(name) for name in ("numpy", "pandas", "scanpy", "anndata")}
    code = {"data.py": file_sha256(data.__file__), "v2_prepare.py": file_sha256(__file__)}
    folds = []
    for seed in args.seeds:
        train, test = split_donor_ids(a.obs["donor_id"], seed=seed)
        fold = prepare_donor_fold(a, train, test, args.n_hvg, args.min_genes, args.min_cells)
        fold.uns["biocellai_fold_seed"] = seed
        fold.uns["biocellai_source_sha256"] = source["sha256"]
        path = out / f"fold_s{seed}.h5ad"
        fold.write_h5ad(path)
        manifest = {
            "protocol": "v2", "seed": seed, "source": source,
            "environment": environment, "code_sha256": code,
            "train_donors": sorted(train.tolist()), "test_donors": sorted(test.tolist()),
            "n_cells": fold.n_obs, "n_genes": fold.n_vars,
            "cells_per_donor": fold.obs["donor_id"].astype(str).value_counts().to_dict(),
            "obs_names_sha256": ordered_ids_sha256(fold.obs_names),
            "var_names_sha256": ordered_ids_sha256(fold.var_names),
            "gene_names": fold.var_names.astype(str).tolist(),
            "matrix_sha256": file_sha256(path),
            "preprocessing": {
                "normalization": fold.uns["biocellai_normalization"],
                "hvg_fit": fold.uns["biocellai_hvg_fit"],
                "n_hvg_requested": args.n_hvg, "min_genes": args.min_genes,
                "min_cells_train": args.min_cells,
            },
            "scientific_status": "prepared_not_evaluated",
        }
        with (out / f"fold_s{seed}.json").open("x") as f:
            json.dump(manifest, f, indent=2)
        folds.append({"seed": seed, "matrix": path.name, "manifest": f"fold_s{seed}.json"})
        print(f"seed={seed} cells={fold.n_obs} genes={fold.n_vars} "
              f"train_donors={len(train)} test_donors={len(test)}", flush=True)
    with (out / "PREPARED.json").open("x") as f:
        json.dump({"protocol": "v2", "folds": folds,
                   "scientific_results_complete": False}, f, indent=2)


if __name__ == "__main__":
    main()
