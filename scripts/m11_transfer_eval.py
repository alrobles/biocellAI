#!/usr/bin/env python3
"""M11-A2 — zero-shot cross-cohort transfer evaluation.

Applies a trained biocellai encoder (saved by experiment m2 --save-embeddings as
model_s{seed}_{cm}_{tex}.pt) to a target dataset already projected onto the
model's gene space, pools cells per donor, and evaluates the donor-level
pathology axis WITHOUT fitting anything on the target cohort:

  - PC1 trajectory correlation: PCA over donor embeddings; sign oriented on
    a designated orientation half of donors, Spearman rho reported on the
    held-out half (honest, matches trajectory_correlation).
  - bootstrap CI over held-out donors (resample B=1000).

Directions enabled by the shared SEA-AD gene space:
  SEA-AD->ROSMAP : m9_adnc model -> rosmap_for_seaad.h5ad vs path_level
  ROSMAP->SEA-AD : rosmap_path model (trained on rosmap_for_seaad)
                   -> seaad_allregions_s0.h5ad vs CPS_Global / adnc
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from biocellai.model import CellEncoder
from biocellai.progression import donor_embeddings, ordinal_map
from biocellai.train import _to_tensor


def _load_model(pt_path: str, device: str):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    enc = CellEncoder(len(ck["var_names"]), tuple(ck["hidden"]),
                      ck["embed_dim"], ck["dropout"]).to(device)
    enc.load_state_dict(ck["enc"])
    sd = ck["proj_p"] if ck.get("proj_p") is not None else ck["proj"]
    out_dim = sd["weight"].shape[0]
    proj = torch.nn.Linear(ck["embed_dim"], out_dim).to(device)
    proj.load_state_dict(sd)
    enc.eval(); proj.eval()
    return enc, proj, ck["var_names"]


def _embed(enc, proj, X, device, bs=4096):
    X = _to_tensor(X)
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            outs.append(proj(enc(X[i:i + bs].to(device))).cpu().numpy())
    return np.concatenate(outs)


def _traj_rho(demb: pd.DataFrame, y: pd.Series, tr_donors, te_donors):
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    y = y.loc[demb.index]
    ok = y.notna()
    demb, y = demb[ok], y[ok]
    tr = [d for d in tr_donors if d in demb.index]
    te = [d for d in te_donors if d in demb.index]
    if len(te) < 5 or len(tr) < 5:
        return np.nan, np.nan, np.array([])
    pc1 = PCA(n_components=1, random_state=0).fit_transform(demb.values)[:, 0]
    pc1 = pd.Series(pc1, index=demb.index)
    # orient sign on the orientation half
    if spearmanr(pc1.loc[tr], y.loc[tr]).statistic < 0:
        pc1 = -pc1
    r = spearmanr(pc1.loc[te], y.loc[te]).statistic
    # bootstrap CI over held-out donors
    rng = np.random.default_rng(0)
    boots = []
    te_arr = np.asarray(te)
    for _ in range(1000):
        b = rng.choice(te_arr, len(te_arr), replace=True)
        if len(np.unique(y.loc[b])) < 2:
            continue
        boots.append(spearmanr(pc1.loc[b], y.loc[b]).statistic)
    lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if boots
              else (np.nan, np.nan))
    return r, (lo, hi), pc1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--target-h5ad", required=True)
    p.add_argument("--donor-table", required=True)
    p.add_argument("--target", required=True,
                   help="donor-table column (continuous) to correlate")
    p.add_argument("--ordinal-col", default=None,
                   help="obs/donor col to map through --ordinal-map for a "
                        "second trajectory (e.g. pathology/adnc)")
    p.add_argument("--ordinal-map", default=None,
                   help="JSON {label: level}")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adata = ad.read_h5ad(args.target_h5ad)
    donors = pd.read_csv(args.donor_table, index_col=0)
    donors_g = donors[~donors.index.duplicated(keep="first")]
    y_main = donors_g[args.target].astype(float)
    y_ord = None
    if args.ordinal_col and args.ordinal_map:
        omap = {k: float(v) for k, v in
                json.loads(Path(args.ordinal_map).read_text()).items()}
        y_ord = ordinal_map(donors_g[args.ordinal_col], omap)

    rows = []
    for f in sorted(glob.glob(str(Path(args.model_dir) / "model_*.pt"))):
        m = re.match(r"model_s(\d+)_(.*?)_((?:all-)?MiniLM-L6-v2|SapBERT-.*)\.pt",
                     Path(f).name)
        if not m:
            continue
        seed, cm, tex = int(m.group(1)), m.group(2), m.group(3)
        enc, proj, var_names = _load_model(f, device)
        # align target X to the model's gene space (order + zero-pad missing)
        tv_set = set(map(str, adata.var_names))
        keep = [i for i, g in enumerate(var_names) if g in tv_set]
        if len(keep) < len(var_names):
            print(f"warn: {len(var_names) - len(keep)} model genes absent "
                  f"from target -> zero-padded", flush=True)
        pos = adata.var_names.get_indexer([var_names[i] for i in keep])
        aligned = np.zeros((adata.n_obs, len(var_names)), dtype=np.float32)
        sub = adata.X[:, pos]
        aligned[:, keep] = (sub.toarray() if hasattr(sub, "toarray")
                            else np.asarray(sub))
        emb = _embed(enc, proj, aligned, device, args.batch_size)
        demb = donor_embeddings(emb, adata.obs["donor_id"])

        # orientation/eval split: donors not seen by the SOURCE model are
        # all eval-eligible; use half for PC1 sign orientation, rest held-out
        rng = np.random.default_rng(seed)
        all_d = np.array(demb.index)
        perm = rng.permutation(all_d)
        half = len(perm) // 2
        tr_d, te_d = perm[:half], perm[half:]

        for tname, y in ((args.target, y_main), ("ordinal", y_ord)):
            if y is None:
                continue
            r, ci, pc1 = _traj_rho(demb, y, tr_d, te_d)
            rows.append(dict(model=Path(f).stem, seed=seed, arm=f"{cm}_{tex}",
                             target=tname, n_te=len(te_d), rho=r,
                             ci_lo=ci[0] if isinstance(ci, tuple) else np.nan,
                             ci_hi=ci[1] if isinstance(ci, tuple) else np.nan))
            print(f"{Path(f).stem} {tname}: rho={r:.3f} "
                  f"CI[{ci[0]:.3f},{ci[1]:.3f}] n_te={len(te_d)}",
                  flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "transfer_metrics.csv", index=False)


if __name__ == "__main__":
    main()
