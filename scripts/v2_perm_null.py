"""Permutation-null replicates for the v2 matrix (ADR-011 R6).

One invocation = one shared donor->pathology permutation applied to every
requested arm:

  B2   trains the dual-head encoder on PERMUTED training labels (the N1
       supervision null), then fits the donor ridge on the TRUE target —
       replicates N1 with perm index k.
  B0/B1 features are label-free; the donor ridge is fit on the permuted
       train-donor targets.
  T*/N2 encoder is frozen from the run checkpoint; the donor ridge is fit
       on the permuted train-donor targets.

The permutation map (perm_seed = seed*100003 + perm_index) is shared across
arms, as the spec requires the same maps for the whole MLP matrix.

Usage:
    python scripts/v2_perm_null.py --fold fold_s0.h5ad \
        --run run_s0_full --out perm_null_s0 --perm-index 0 \
        --label-col cell_type --pathology-col cps_global
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from biocellai import train as _train, v2run as _v2run  # noqa: E402
from biocellai.progression import (  # noqa: E402
    donor_embeddings, pseudobulk_donor,
)
from biocellai.revalidation import file_sha256  # noqa: E402
from biocellai.transfer import encode_cells, load_encoder  # noqa: E402
from biocellai.train import encode_labels  # noqa: E402
from biocellai.v2run import composition_features, donor_target  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def permuted_target(target: pd.Series, train_donors, perm_seed: int):
    """Same donor->label map for every arm: values reshuffled among the
    TRAIN donors only; test donors keep their true values."""
    rng = np.random.default_rng(perm_seed)
    vals = target.loc[list(train_donors)].to_numpy()
    out = target.copy()
    out.loc[list(train_donors)] = vals[rng.permutation(len(vals))]
    return out


def _ridge_rho(feats: pd.DataFrame, target: pd.Series, train_donors,
               test_donors, alpha: float) -> float:
    """Ridge fit on train donors of `target` (possibly permuted) ->
    Spearman rho of test-donor predictions vs their TRUE labels."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(feats.loc[list(train_donors)].to_numpy(),
              target.loc[list(train_donors)].to_numpy())
    pred = pd.Series(
        model.predict(feats.loc[list(test_donors)].to_numpy()),
        index=pd.Index(test_donors))
    obs = target.loc[list(test_donors)]
    if np.ptp(obs.to_numpy()) == 0 or np.ptp(pred.to_numpy()) == 0:
        return np.nan
    return float(spearmanr(obs, pred).statistic)


def _train_config(man: dict, device: str) -> _train.TrainConfig:
    kw = {}
    for k, v in man["config"]["train"].items():
        if k == "hidden":
            v = ast.literal_eval(v) if isinstance(v, str) else v
        elif k in ("lr", "dropout"):
            v = float(v)
        elif str(v).lstrip("-").isdigit():
            v = int(v)
        kw[k] = v
    kw["device"] = device
    return _train.TrainConfig(**kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True,
                   help="completed v2 run dir (ckpts + manifest)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--perm-index", type=int, required=True)
    p.add_argument("--arms", default="B0,B1,B2,T0,T1,T2,T3,N2")
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--pathology-col", required=True)
    p.add_argument("--covariates", default="")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    fold = ad.read_h5ad(args.fold)
    man = json.loads((args.run / "manifest.json").read_text())
    seed = int(man["seed"])
    tc = _train_config(man, args.device)
    perm_seed = seed * 100003 + args.perm_index

    is_test = fold.obs["is_test"].to_numpy()
    donors = fold.obs["donor_id"].astype(str)
    train_donors = sorted(donors[~is_test].unique())
    test_donors = sorted(donors[is_test].unique())
    target = donor_target(fold.obs, args.pathology_col)
    ptarget = permuted_target(target, train_donors, perm_seed)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    rows = []

    def record(arm, rho):
        rows.append({"arm": arm, "perm": args.perm_index,
                     "perm_seed": perm_seed, "rho": rho,
                     "n_train_donors": len(train_donors),
                     "n_test_donors": len(test_donors)})

    # --- B2: encoder retrained on permuted labels (N1 replicate) ---------
    y, _cats = encode_labels(fold, args.label_col)
    obs2_path = donors.map(ptarget)
    enc, _h_t, _h_p, _ = _v2run.train_supervised_dual(
        fold.X[~is_test], y[~is_test],
        obs2_path.to_numpy(dtype=np.float32)[~is_test], tc)
    emb = encode_cells(enc, fold.X, device=args.device)
    demb = donor_embeddings(emb, donors)
    record("B2", _ridge_rho(demb, target, train_donors, test_donors,
                            args.ridge_alpha))

    # --- readout-level nulls on frozen features --------------------------
    for arm in arms:
        if arm == "B2":
            continue
        if arm == "B0":
            feats = composition_features(
                fold, args.label_col, ~is_test,
                tuple(c for c in args.covariates.split(",") if c))
        elif arm == "B1":
            feats = pseudobulk_donor(fold.X, donors, seed=seed,
                                     train_donors=train_donors)
        else:
            ckpt = args.run / f"ckpt_{arm}.pt"
            if not ckpt.exists():
                continue
            enc = load_encoder(ckpt, fold.n_vars, tc.hidden,
                               tc.embed_dim, device=args.device)
            feats = donor_embeddings(
                encode_cells(enc, fold.X, device=args.device), donors)
        record(arm, _ridge_rho(feats, ptarget, train_donors, test_donors,
                               args.ridge_alpha))

    # shared across array tasks: create once, refuse to overwrite this perm
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out
    perm_file = out / f"perm_{args.perm_index}.csv"
    if perm_file.exists():
        raise ValueError(f"{perm_file} exists — refusing to overwrite")
    df = pd.DataFrame(rows)
    df.to_csv(perm_file, index=False)
    if not (out / "manifest.json").exists():
        (out / "manifest.json").write_text(json.dumps({
            "protocol": "v2_perm_null", "perm_index": args.perm_index,
            "perm_seed": perm_seed,
            "fold": {"path": str(args.fold.resolve()),
                     "sha256": file_sha256(args.fold)},
            "run": str(args.run.resolve()),
            "arms": [r["arm"] for r in rows],
            "design": ("shared train-donor->label permutation per index; "
                       "B2 retrains encoder on permuted labels (N1 null); "
                       "other arms permute the ridge target on frozen "
                       "features")
        }, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
