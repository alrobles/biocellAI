#!/usr/bin/env python3
"""M11-A — build ROSMAP Liu2025 multi-region dataset for the biocellai pipeline.

Input : snRNA_multiregion_liu2025.h5ad (2.26M nuclei, 111 donors, 6 regions,
        open processed release — Mathys/Xiong/Liu integration).
Output: rosmap_s0.h5ad   — subsampled, standardised obs (donor_id, region,
                           cell_type, pathology, path_level) + log-normalised
        rosmap_donors.csv — donor-level table (pathology level, covariates)

Sampling: up to --max-cells-per-donor-region cells per donor×region keeps all
six regions represented while bounding memory (~111×6×800 ≈ 500k nuclei).

Pathology mapping (pre-registered): nonAD=0, earlyAD=1, lateAD=2. The open
release carries only this 3-level label — no Braak/CERAD/cogdx — so the
replication target is the ordinal pathology level, documented in the ADR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

PATH_MAP = {"nonAD": 0, "earlyAD": 1, "lateAD": 2}
KEEP_OBS = ["ROSMAP_IndividualID", "BrainRegion", "RNA.Class", "RNA.Subclass",
            "RNA.Subtype", "Pathology", "age_death", "Sex", "pmi", "Study",
            "Sample", "n_genes_by_counts", "pct_counts_mt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="rosmap_s0.h5ad")
    ap.add_argument("--donor-table", default="rosmap_donors.csv")
    ap.add_argument("--max-cells-per-donor-region", type=int, default=800)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    a = ad.read_h5ad(args.input, backed="r")
    obs = a.obs
    dr = obs["ROSMAP_IndividualID"].astype(str) + "|" \
        + obs["BrainRegion"].astype(str)
    groups = dr.unique()
    keep = []
    for g in groups:
        idx = np.where(dr == g)[0]
        if len(idx) > args.max_cells_per_donor_region:
            idx = np.sort(rng.choice(idx, args.max_cells_per_donor_region,
                                     replace=False))
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    print(f"subsampled {len(keep)} / {a.n_obs} cells "
          f"({len(groups)} donor×region groups)", flush=True)

    # materialise subset into memory
    sub = a[keep].to_memory()
    sub.obs = sub.obs[[c for c in KEEP_OBS if c in sub.obs.columns]].copy()

    sub.obs["donor_id"] = sub.obs["ROSMAP_IndividualID"].astype(str)
    sub.obs["region"] = sub.obs["BrainRegion"].astype(str)
    sub.obs["cell_type"] = sub.obs["RNA.Subclass"].astype(str)
    sub.obs["pathology"] = sub.obs["Pathology"].astype(str)
    sub.obs["path_level"] = sub.obs["pathology"].map(PATH_MAP).astype("Int64")

    # donor-level table (pathology assumed donor-constant; verify + warn)
    dt = (sub.obs.groupby("donor_id")
          .agg(n_cells=("donor_id", "size"),
               n_regions=("region", "nunique"),
               pathology=("pathology", lambda s: s.mode().iloc[0]),
               path_level=("path_level", "median"),
               pathology_uniq=("pathology", "nunique"),
               age_death=("age_death", "first"),
               Sex=("Sex", "first"),
               pmi=("pmi", "median")))
    n_var = int((dt["pathology_uniq"] > 1).sum())
    print(f"donors with >1 pathology label across regions: {n_var}", flush=True)
    dt.drop(columns="pathology_uniq").to_csv(args.donor_table)
    print(f"donor table -> {args.donor_table} ({len(dt)} donors)", flush=True)

    # normalize like the SEA-AD pipeline (counts -> log1p CPM-ish -> HVG)
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sc.pp.highly_variable_genes(sub, n_top_genes=args.n_hvg, flavor="seurat",
                                subset=False)
    sub = sub[:, sub.var.highly_variable].copy()

    sub.write_h5ad(args.out)
    manifest = dict(
        n_cells=int(sub.n_obs), n_genes=int(sub.n_vars),
        n_donors=int(sub.obs.donor_id.nunique()),
        n_regions=int(sub.obs.region.nunique()),
        regions=sorted(sub.obs.region.unique().tolist()),
        cell_types=sorted(sub.obs.cell_type.unique().tolist()),
        pathology_counts=sub.obs.pathology.value_counts().to_dict(),
        donors_with_mixed_pathology=n_var,
        max_cells_per_donor_region=args.max_cells_per_donor_region,
        n_hvg=args.n_hvg, seed=args.seed)
    Path(str(args.out) + ".manifest.json").write_text(json.dumps(manifest,
                                                                 indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
