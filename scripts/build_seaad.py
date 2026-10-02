#!/usr/bin/env python3
"""M7 — build the SEA-AD MTG+V1C working dataset.

Loads the official Allen h5ad per region (backed), stratified-subsamples
by (donor, Subclass), joins CPS, concatenates on shared genes, runs the
standard preprocess (normalize+log1p+HVG), writes:
  - data/raw/seaad_mtg_v1c_{n}_s{seed}.h5ad
  - data/manifests/seaad_donors.csv   (per-donor pathology + CPS)
  - data/manifests/seaad_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from biocellai.captions import marker_table
from biocellai.data import (DatasetManifest, donor_split, load_seaad_region,
                        preprocess, use_gene_symbols)

DATA = Path("/beegfs/a474r867/biocellai/data")
REL = "2026-06-22"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA))
    p.add_argument("--max-cells", type=int, default=150_000)
    p.add_argument("--regions", nargs="+", default=["MTG", "V1C"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--donor-out", default=None,
                   help="donor table path (default: <data-dir>/seaad_donors.csv)")
    p.add_argument("--markers-out", default=None,
                   help="marker manifest path (default: "
                        "data/manifests/markers_seaad_s0.json)")
    p.add_argument("--manifest-out", default=None,
                   help="dataset manifest path (default: "
                        "<data-dir>/seaad_manifest.json)")
    args = p.parse_args()
    d = Path(args.data_dir)

    adatas, donors = [], []
    for region in args.regions:
        h5 = d / f"SEAAD_{region}_RNAseq_final-nuclei.{REL}.h5ad"
        sub, dt = load_seaad_region(
            h5, region, cps_csv=d / "Global_and_Local_CPS.20260105.csv",
            max_cells=args.max_cells // len(args.regions), seed=args.seed)
        sub.obs["region"] = region
        dt["region"] = region
        adatas.append(sub)
        donors.append(dt)
        print(f"{region}: {sub.n_obs} cells, {sub.obs['donor_id'].nunique()} donors, "
              f"{sub.obs['cell_type'].nunique()} subclasses", flush=True)

    # keep only the columns used downstream — the ~90 clinical/demographic
    # obs fields slow concat and make h5ad writing pathological
    KEEP = ["donor_id", "cell_type", "region", "adnc", "braak", "cerad",
            "thal", "cognitive_status", "casi", "apoe", "sex",
            "age_at_death", "n_genes", "CPS_Global", "CPS_Global_ABeta",
            "CPS_Global_pTau", "CPS_Local", "CPS_Local_ABeta",
            "CPS_Local_pTau"]
    for a in adatas:
        keep = [c for c in KEEP if c in a.obs.columns]
        a.obs = a.obs[keep]

    print("concatenating...", flush=True)
    adata = ad.concat(adatas, join="inner", label=None, index_unique=None)
    # h5py keys cannot contain '/' (e.g. "Race (choice=Black/ African American)")
    adata.obs.columns = [c.replace("/", "_") for c in adata.obs.columns]
    adata.var.columns = [c.replace("/", "_") for c in adata.var.columns]
    # mixed-type obs columns across regions (e.g. str + bool/NaN in
    # 'Neurotypical reference') fail h5py's vlen-str write — coerce to str
    for c in adata.obs.columns:
        if adata.obs[c].dtype == object:
            adata.obs[c] = adata.obs[c].astype(str)
    adata = use_gene_symbols(adata) if "feature_name" in adata.var.columns else adata
    print("preprocess...", flush=True)
    adata = preprocess(adata)

    donor_table = pd.concat(donors)
    donor_table.to_csv(args.donor_out or (d / "seaad_donors.csv"))
    print("donor table:", len(donor_table), "rows", flush=True)

    out = args.out or str(d / f"seaad_{'_'.join(r.lower() for r in args.regions)}_"
                            f"{adata.n_obs}_s{args.seed}.h5ad")
    print("writing", out, flush=True)
    adata.write_h5ad(out)
    print("write done", flush=True)

    manifest = DatasetManifest(
        name="seaad_" + "_".join(args.regions),
        source="sea-ad-single-cell-profiling s3 (official release " + REL + ")",
        filters={"regions": args.regions, "max_cells": args.max_cells},
        n_cells=int(adata.n_obs), n_genes_hvg=int(adata.n_vars),
        n_donors=int(adata.obs["donor_id"].nunique()),
        n_cell_types=int(adata.obs["cell_type"].nunique()),
        cells_per_donor=adata.obs["donor_id"].value_counts().to_dict(),
        cells_per_type=adata.obs["cell_type"].value_counts().to_dict(),
        seed=args.seed, date=pd.Timestamp.now().isoformat())
    manifest.write(args.manifest_out or (d / "seaad_manifest.json"))

    # train-donor marker table (label-derived query source for M7 retrieval)
    tr_mask, _ = donor_split(adata, seed=args.seed)
    markers = marker_table(adata[tr_mask], n_markers=8)
    mpath = Path(args.markers_out or "data/manifests/markers_seaad_s0.json")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(markers, indent=2))
    print("wrote", out, "| donors:", len(donor_table),
          "| markers →", mpath)


if __name__ == "__main__":
    main()
