"""Extract a verified-counts AnnData with a normalized obs schema for
v2_prepare (ADR-011 R1-REAL-DATA).

Raw cohort releases differ in schema: SEA-AD stores raw UMIs in a layer and
donors in 'Donor ID'; ROSMAP Liu2025 keeps counts in X and donors in
'ROSMAP_IndividualID'. This script produces a uniform input: X = verified
counts, obs['donor_id'], optional donor-table join (pathology targets),
optional per-donor cell cap. Writes <out>.h5ad + <out>.extract.json.

Example (SEA-AD MTG):
    python scripts/v2_extract.py \
        --input data/SEAAD_MTG_RNAseq_final-nuclei.2026-06-22.h5ad \
        --counts-layer UMIs --donor-col "Donor ID" \
        --keep-obs "Donor ID,Subclass,Class,Cognitive Status,Braak,CERAD score,Brain Region" \
        --donor-table data/seaad_donors.csv --donor-table-key donor_id \
        --max-cells-per-donor 1500 --seed 0 --out data/v2_src/seaad_mtg
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from biocellai.data import validate_counts
from biocellai.revalidation import file_sha256, ordered_ids_sha256


def _csr_rows(X, rows):
    """Vectorized CSR row-gather; scipy fancy indexing crawls at 1e9+ nnz."""
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    rows = np.asarray(rows, dtype=np.int64)
    indptr = X.indptr.astype(np.int64, copy=False)
    nnz = indptr[rows + 1] - indptr[rows]
    new_indptr = np.empty(len(rows) + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(nnz, out=new_indptr[1:])
    # destination position p in row k maps to source indptr[k] + (p - indptr[k])
    gather = np.repeat(indptr[rows] - new_indptr[:-1], nnz) + np.arange(
        new_indptr[-1], dtype=np.int64)
    return sparse.csr_matrix(
        (X.data[gather], X.indices[gather], new_indptr),
        shape=(len(rows), X.shape[1]), dtype=X.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--counts-layer", required=True,
                   help="X or the verified raw-count layer name")
    p.add_argument("--donor-col", required=True,
                   help="obs column holding donor identity; renamed to donor_id")
    p.add_argument("--keep-obs", default="",
                   help="comma-separated obs columns to keep (donor col always kept)")
    p.add_argument("--rename-obs", action="append", default=[], metavar="SRC=DST",
                   help="rename an obs column; repeatable")
    p.add_argument("--donor-table", type=Path, default=None,
                   help="CSV with donor-level columns to join onto obs")
    p.add_argument("--donor-table-key", default="donor_id")
    p.add_argument("--max-cells-per-donor", type=int, default=0,
                   help="0 = keep all cells")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True,
                   help="output prefix; writes <out>.h5ad and <out>.extract.json")
    args = p.parse_args()

    src_sha = file_sha256(args.input)
    a = ad.read_h5ad(args.input)
    var_index = a.var_names.astype(str)

    keep = {args.donor_col}
    keep.update(c.strip() for c in args.keep_obs.split(",") if c.strip())
    obs = a.obs[list(keep & set(a.obs.columns))].copy()
    missing = keep - set(a.obs.columns)
    if missing:
        raise ValueError(f"obs columns not found: {sorted(missing)}")
    obs["donor_id"] = obs[args.donor_col].astype(str)
    for spec in args.rename_obs:
        src, _, dst = spec.partition("=")
        if src in obs.columns:
            obs[dst.strip()] = obs.pop(src.strip())

    counts = a.X if args.counts_layer == "X" else a.layers[args.counts_layer]
    del a  # release obs/uns; X itself is never mutated downstream
    out = ad.AnnData(X=counts, obs=obs,
                     var=pd.DataFrame(index=var_index))
    del counts  # drop the alias; out.X keeps the matrix alive

    if args.donor_table is not None:
        table = pd.read_csv(args.donor_table)
        table[args.donor_table_key] = table[args.donor_table_key].astype(str)
        table = table.drop_duplicates(subset=args.donor_table_key).set_index(
            args.donor_table_key)
        add = [c for c in table.columns if c not in out.obs.columns]
        for col in add:
            out.obs[col] = out.obs["donor_id"].map(table[col])
        n_missing = int(out.obs["donor_id"].isin(set(table.index)).eq(False).sum())
        if n_missing:
            raise ValueError(f"{n_missing} cells have donors absent from "
                             f"--donor-table; fix mapping before proceeding")
    # ensure unique names before any subsampling
    if not out.obs_names.is_unique:
        raise ValueError("cell identifiers are not unique in source")
    if not out.var_names.is_unique:
        out.var_names_make_unique()

    if args.max_cells_per_donor > 0:
        rng = np.random.default_rng(args.seed)
        keep_idx = []
        grouped = out.obs.groupby("donor_id").indices
        for idx in grouped.values():
            if len(idx) > args.max_cells_per_donor:
                idx = rng.choice(idx, size=args.max_cells_per_donor,
                                 replace=False)
            keep_idx.append(idx)
        keep_idx = np.sort(np.concatenate(keep_idx))
        if sparse.issparse(out.X):
            out = ad.AnnData(X=_csr_rows(out.X, keep_idx),
                             obs=out.obs.iloc[keep_idx].copy(),
                             var=out.var)
        else:
            out = out[keep_idx].copy()

    validate_counts(out)  # fails loudly if the source is not integer counts
    out.uns["biocellai_data_state"] = "counts"

    h5_path = args.out.with_suffix(".h5ad")
    meta_path = args.out.with_suffix(".extract.json")
    for f in (h5_path, meta_path):
        if f.exists():
            raise FileExistsError(f"refusing to overwrite {f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(h5_path)
    manifest = {
        "step": "v2_extract", "seed": args.seed,
        "source": {"path": str(args.input.resolve()), "sha256": src_sha,
                   "counts_layer": args.counts_layer},
        "donor_col": args.donor_col,
        "rename_obs": args.rename_obs,
        "donor_table": (str(args.donor_table.resolve())
                        if args.donor_table else None),
        "max_cells_per_donor": args.max_cells_per_donor,
        "n_cells": out.n_obs, "n_genes": out.n_vars,
        "n_donors": int(out.obs["donor_id"].nunique()),
        "cells_per_donor": out.obs["donor_id"].value_counts().to_dict(),
        "obs_names_sha256": ordered_ids_sha256(out.obs_names),
        "var_names_sha256": ordered_ids_sha256(out.var_names),
        "output_sha256": file_sha256(h5_path),
        "environment": {n: version(n) for n in
                        ("numpy", "pandas", "anndata")},
        "scientific_status": "extracted_not_evaluated",
    }
    meta_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {h5_path} cells={out.n_obs} donors={manifest['n_donors']} "
          f"genes={out.n_vars}")


if __name__ == "__main__":
    main()
