"""G2 — scGPT whole-human zero-shot embeddings on the SEA-AD split.

Uses the official whole-human checkpoint (data/scgpt_whole_human:
best_model.pt + vocab.json + args.json) and scgpt.tasks.embed_data
(official binning + embedding pipeline). Gene col = var symbols.

Eval: identical protocol as our arms — cell-type probe macro-F1 and
donor-pooled ridge CPS_Global rho on held-out donors, 3 seeds, plus
ordinal trajectories. Same donor_split as all other arms.

Usage: python scripts/g2_scgpt.py [--max-cells N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from biocellai.data import donor_split
from biocellai.progression import (ADNC_ORDER, BRAAK_ORDER, CERAD_ORDER,
                               ordinal_map, regress_heldout,
                               trajectory_correlation)

ROOT = Path("/beegfs/a474r867/biocellai")
MODEL_DIR = ROOT / "data/scgpt_whole_human"
DATA = ROOT / "data/seaad_allregions_s0.h5ad"
DONORS = ROOT / "data/seaad_donors_allregions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/g2_scgpt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-cells", type=int, default=0)
    args = ap.parse_args()

    import scgpt as scg
    import json
    import torch

    # anndata>=0.10 dropped the dtype kwarg that scgpt 0.2.4 passes
    _orig_adata_init = ad.AnnData.__init__

    def _adata_init_compat(self, X=None, **kw):
        kw.pop("dtype", None)
        _orig_adata_init(self, X=X, **kw)

    ad.AnnData.__init__ = _adata_init_compat

    print("loading", DATA, flush=True)
    adata = ad.read_h5ad(DATA)
    vocab = json.load(open(MODEL_DIR / "vocab.json"))
    keep = np.array([g in vocab for g in adata.var_names.astype(str)])
    print(f"genes: {adata.n_vars} -> in vocab {keep.sum()}", flush=True)
    adata = adata[:, keep].copy()
    adata.var["gene_name"] = adata.var_names.astype(str)

    if args.max_cells:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(adata.n_obs, args.max_cells, replace=False))
        adata = adata[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("embed_data on", device, flush=True)
    emb_adata = scg.tasks.embed_data(
        adata,
        MODEL_DIR,
        gene_col="gene_name",
        batch_size=args.batch,
        device=device,
        return_new_adata=True,
    )
    # scgpt>=0.2 stores the embedding in .X; older releases used .X_emb
    emb = np.asarray(getattr(emb_adata, "X_emb", None)
                     if hasattr(emb_adata, "X_emb") else emb_adata.X)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    donors = adata.obs["donor_id"].astype(str).to_numpy()
    ctypes = adata.obs["cell_type"].astype(str).to_numpy()
    np.savez_compressed(out / "scgpt_emb.npz", emb=emb, donor=donors,
                        cell_type=ctypes,
                        region=adata.obs["region"].astype(str).to_numpy())

    donor_cps = pd.read_csv(DONORS)
    by_donor = donor_cps.groupby("donor_id")
    cps = by_donor["CPS_Global"].first()
    ordinals = {
        "braak": ordinal_map(by_donor["braak"].first(), BRAAK_ORDER),
        "adnc": ordinal_map(by_donor["adnc"].first(), ADNC_ORDER),
        "cerad": ordinal_map(by_donor["cerad"].first(), CERAD_ORDER),
    }

    rows = []
    demb = pd.DataFrame(emb)
    demb["donor"] = donors
    demb = demb.groupby("donor").mean()
    for seed in args.seeds:
        tr_mask, te_mask = donor_split(adata, seed=seed)
        tr_d = pd.Index(pd.unique(donors[tr_mask]))
        te_d = pd.Index(pd.unique(donors[te_mask]))
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, f1_score
        clf = LogisticRegression(max_iter=300)
        clf.fit(emb[tr_mask], ctypes[tr_mask])
        pred = clf.predict(emb[te_mask])
        rows.append(dict(seed=seed, metric="celltype_probe_macroF1",
                         value=f1_score(ctypes[te_mask], pred,
                                        average="macro")))
        rows.append(dict(seed=seed, metric="celltype_probe_balacc",
                         value=balanced_accuracy_score(ctypes[te_mask], pred)))
        r = regress_heldout(demb, cps, tr_d, te_d)
        rows.append(dict(seed=seed, metric="donor_CPS_Global_rho",
                         value=r["rho"]))
        rows.append(dict(seed=seed, metric="donor_CPS_Global_r2",
                         value=r["r2"]))
        for key, t in ordinals.items():
            tr = trajectory_correlation(demb, t, set(tr_d))
            rows.append(dict(seed=seed, metric=f"{key}_trajectory_rho",
                             value=tr["rho"]))

    df = pd.DataFrame(rows)
    df.to_csv(out / "g2_scgpt_metrics.csv", index=False)
    print(df.groupby("metric")["value"].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
