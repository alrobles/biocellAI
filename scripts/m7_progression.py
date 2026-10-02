#!/usr/bin/env python3
"""M7 — donor-level progression evaluation (G5a/G5b/G5c).

Consumes cell-embedding npz files written by `experiment m2
--save-embeddings`, pools them into donor embeddings, and evaluates:

  G5a  ridge regression on CPS_Global / CPS_Local (held-out donors)
       vs pseudobulk-PCA baseline
  G5b  PC1 trajectory rank-correlation vs Braak ordinal
  G5c  MTG-trained embeddings applied to V1C cells of the same donors
       (regional transfer / vulnerability control)

Outputs experiments/m7_progression/progression_metrics.csv.
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
from biocellai.progression import (BRAAK_ORDER, CERAD_ORDER, ADNC_ORDER,
                               donor_embeddings, ordinal_map,
                               pseudobulk_donor, regress_heldout,
                               trajectory_correlation)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--donor-table", required=True, help="seaad_donors.csv")
    p.add_argument("--h5ad", default=None, help="dataset h5ad for pseudobulk baseline")
    p.add_argument("--out", default="experiments/v2_revalidation/progression")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    out = reserve_output(args.out)

    donors = pd.read_csv(args.donor_table, index_col=0)
    # donor table is one row per (donor, region): global pathology cols are
    # identical across rows; CPS_Local is region-specific.
    donors_g = donors[~donors.index.duplicated(keep="first")]
    braak_ord = ordinal_map(donors_g["braak"], BRAAK_ORDER)
    cerad_ord = ordinal_map(donors_g["cerad"], CERAD_ORDER)
    adnc_ord = ordinal_map(donors_g["adnc"], ADNC_ORDER)

    rows = []

    # pseudobulk baseline per seed (expression-only, same donor splits)
    if args.h5ad:
        import anndata as ad
        adata = ad.read_h5ad(args.h5ad)
        for seed in args.seeds:
            tr, te = split_donor_ids(adata.obs["donor_id"], seed=seed)
            emb = pseudobulk_donor(adata.X, adata.obs["donor_id"], seed=seed,
                                   train_donors=tr)
            for target in ("CPS_Global",):
                r = regress_heldout(emb, donors_g[target], tr, te)
                rows.append(dict(arm="pseudobulk_pca", seed=seed,
                                 target=target, **r))
            r = trajectory_correlation(emb, braak_ord,
                                       train_donors=set(tr))
            rows.append(dict(arm="pseudobulk_pca", seed=seed,
                             target="braak_trajectory", **r))

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
        for target in ("CPS_Global", "CPS_Global_pTau", "CPS_Global_ABeta"):
            if target in donors_g.columns:
                r = regress_heldout(demb, donors_g[target], tr_donors, te_donors)
                rows.append(dict(arm=arm, seed=seed, target=target, **r))
        for name, ordinal in (("braak_trajectory", braak_ord),
                              ("cerad_trajectory", cerad_ord),
                              ("adnc_trajectory", adnc_ord)):
            r = trajectory_correlation(demb, ordinal,
                                       train_donors=set(tr_donors))
            rows.append(dict(arm=arm, seed=seed, target=name, **r))

        # G5c: within-region progression — does the MTG signal hold in V1C?
        for region in np.unique(z["region"]):
            if not region:
                continue
            m_r = z["region"] == region
            if m_r.sum() < 500:
                continue
            demb_r = donor_embeddings(z["emb"][m_r], z["donor"][m_r])
            donors_r = donors[donors["region"] == region]
            for target in ("CPS_Local", "CPS_Global"):
                tgt = donors_r[target] if target == "CPS_Local" else donors_g[target]
                r = regress_heldout(demb_r, tgt, tr_donors, te_donors)
                rows.append(dict(arm=f"{arm}@{region}", seed=seed,
                                 target=target, **r))

    if not rows:
        raise ValueError("no recognized inputs; no metrics produced")
    df = pd.DataFrame(rows)
    df["evaluation_protocol"] = "v2_readout_on_unrevalidated_inputs"
    df.to_csv(out / "progression_metrics.csv", index=False)
    print(df.groupby(["arm", "target"])[["rho"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
