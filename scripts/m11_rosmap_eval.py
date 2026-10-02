#!/usr/bin/env python3
"""M11-A1 — ROSMAP replication eval (donor-level pathology).

Mirror of m7_progression.py but for the ROSMAP Liu2025 dataset: pools cell
embeddings into donor embeddings and evaluates against the 3-level ordinal
pathology label (nonAD=0 < earlyAD=1 < lateAD=2) in rosmap_donors.csv.

Metrics:
  - ridge held-out donor regression on path_level (rho, r2) — analogue of
    the CPS_Global target in SEA-AD
  - PC1 trajectory correlation over the ordinal pathology levels
  - per-region breakdown (6 regions)
  - pseudobulk-PCA baseline on the same splits

Outputs experiments/m11_rosmap/progression_metrics.csv.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from biocellai.data import split_donor_ids
from biocellai.revalidation import reserve_output
from biocellai.progression import (donor_embeddings, ordinal_map,
                               pseudobulk_donor, regress_heldout,
                               trajectory_correlation)

PATH_ORDER = {"nonAD": 0.0, "earlyAD": 1.0, "lateAD": 2.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--donor-table", required=True)
    p.add_argument("--h5ad", default=None)
    p.add_argument("--out", default="experiments/v2_revalidation/rosmap_readout")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    out = reserve_output(args.out)

    donors = pd.read_csv(args.donor_table, index_col=0)
    donors_g = donors[~donors.index.duplicated(keep="first")]
    path_ord = ordinal_map(donors_g["pathology"], PATH_ORDER)
    path_level = donors_g["path_level"].astype(float)

    rows = []

    if args.h5ad:
        import anndata as ad
        adata = ad.read_h5ad(args.h5ad)
        for seed in args.seeds:
            tr, te = split_donor_ids(adata.obs["donor_id"], seed=seed)
            emb = pseudobulk_donor(adata.X, adata.obs["donor_id"], seed=seed,
                                   train_donors=tr)
            r = regress_heldout(emb, path_level, tr, te)
            rows.append(dict(arm="pseudobulk_pca", seed=seed,
                             target="path_level", **r))
            r = trajectory_correlation(emb, path_ord, train_donors=set(tr))
            rows.append(dict(arm="pseudobulk_pca", seed=seed,
                             target="path_trajectory", **r))

    for f in sorted(glob.glob(str(Path(args.emb_dir) / "emb_*.npz"))):
        m = re.match(r"emb_s(\d+)_(.*?)_((?:all-)?MiniLM-L6-v2|SapBERT-.*)\.npz",
                     Path(f).name)
        if not m:
            continue
        seed, mode, tex = int(m.group(1)), m.group(2), m.group(3)
        z = np.load(f, allow_pickle=True)
        demb = donor_embeddings(z["emb"], z["donor"])
        te_mask = z["is_test"].astype(bool)
        te_donors = np.unique(z["donor"][te_mask])
        tr_donors = np.array([d for d in demb.index if d not in set(te_donors)])

        arm = f"{mode}_{tex}"
        r = regress_heldout(demb, path_level, tr_donors, te_donors)
        rows.append(dict(arm=arm, seed=seed, target="path_level", **r))
        r = trajectory_correlation(demb, path_ord, train_donors=set(tr_donors))
        rows.append(dict(arm=arm, seed=seed, target="path_trajectory", **r))

        for region in np.unique(z["region"]):
            if not region:
                continue
            m_r = z["region"] == region
            if m_r.sum() < 500:
                continue
            demb_r = donor_embeddings(z["emb"][m_r], z["donor"][m_r])
            r = regress_heldout(demb_r, path_level, tr_donors, te_donors)
            rows.append(dict(arm=f"{arm}@{region}", seed=seed,
                             target="path_level", **r))

    if not rows:
        raise ValueError("no recognized inputs; no metrics produced")
    df = pd.DataFrame(rows)
    df["evaluation_protocol"] = "v2_readout_on_unrevalidated_inputs"
    df.to_csv(out / "progression_metrics.csv", index=False)
    print(df.groupby(["arm", "target"])[["rho"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
