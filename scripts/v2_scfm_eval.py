"""Evaluate frozen scFM embeddings under the identical v2 protocol.

For each embedding npz (written by v2_scfm_embed.py): linear probe on
train-donor cells -> macro-F1 on test-donor cells; donor-mean embeddings
-> train-fit ridge -> pathology rho/R2/MAE on held-out donors. Outputs a
run directory (metrics.csv + predictions_<arm>.csv + manifest.json)
identical in schema to scripts/v2_run.py, so v2_aggregate/v2_report
consume it directly. Arm names are <model>_<input> (e.g. gf_v2_104m_hvg).

    python scripts/v2_scfm_eval.py \
        --emb "experiments/v2_revalidation/rosmap/scfm/*_s0.npz" \
        --fold experiments/v2_revalidation/rosmap/folds/fold_s0.h5ad \
        --fold-manifest experiments/v2_revalidation/rosmap/folds/fold_s0.json \
        --pathology-col path_level --label-col cell_type \
        --out experiments/v2_revalidation/rosmap/run_s0_scfm
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from biocellai.progression import donor_embeddings, ridge_predictions
from biocellai.revalidation import file_sha256, reserve_output
from biocellai.train import TrainConfig, linear_probe, metrics


def _eval_one(npz_path, fold_obs, train_mask, test_mask, y, target,
              train_donors, test_donors, device):
    z = np.load(npz_path, allow_pickle=False)
    emb = z["emb"].astype(np.float32)
    order = pd.Index(z["obs_names"].astype(str))
    pos = order.get_indexer(fold_obs.index.astype(str))
    present = pos >= 0  # model may drop cells (e.g. empty GF sequences)
    emb = emb[pos[present]]
    tr = train_mask & present
    te = test_mask & present
    cfg = TrainConfig(device=device, probe_epochs=200,
                      embed_dim=int(emb.shape[1]))
    enc = torch.nn.Identity()
    probe_preds = linear_probe(enc, emb[tr[present]], y[tr],
                               emb[te[present]], cfg)
    m = {"identity_f1_probe": float(metrics(y[te], probe_preds)
                                   ["macro_f1"]),
         "identity_f1_zeroshot": np.nan, "identity_eval": "probe",
         "path_rho": np.nan, "path_r2": np.nan, "path_mae": np.nan,
         "n_cells": int(present.sum()),
         "n_missing_cells": int((~present).sum())}
    pred = None
    if target is not None:
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_absolute_error, r2_score
        demb = donor_embeddings(
            emb, fold_obs["donor_id"].astype(str)[present])
        pred = ridge_predictions(demb, target, train_donors, test_donors)
        m["path_rho"] = float(spearmanr(pred.predicted, pred.observed)
                              .statistic)
        m["path_r2"] = float(r2_score(pred.observed, pred.predicted))
        m["path_mae"] = float(mean_absolute_error(pred.observed,
                                                  pred.predicted))
        pred = pred.reset_index().rename(columns={"index": "donor_id"})
    return m, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True,
                   help="glob of npz files for this fold's cells")
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--fold-manifest", type=Path, required=True)
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--pathology-col", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    fold = ad.read_h5ad(args.fold)
    fman = json.loads(args.fold_manifest.read_text())
    train_donors = list(map(str, fman["train_donors"]))
    test_donors = list(map(str, fman["test_donors"]))
    obs = fold.obs
    donors = obs["donor_id"].astype(str)
    train_mask = donors.isin(set(train_donors)).to_numpy()
    test_mask = donors.isin(set(test_donors)).to_numpy()
    cats = pd.Categorical(obs[args.label_col].astype(str))
    y = cats.codes.astype(np.int64)

    target = None
    if args.pathology_col and args.pathology_col in obs.columns:
        per_donor = obs.groupby(donors)[args.pathology_col]
        target = per_donor.first().astype(float)

    npzs = sorted(glob.glob(args.emb))
    if not npzs:
        raise FileNotFoundError(f"--emb glob matched no files: {args.emb}")
    out = reserve_output(args.out)
    rows, preds = [], {}
    for npz in npzs:
        arm = Path(npz).name.rsplit("_s", 1)[0]
        m, pred = _eval_one(npz, obs, train_mask, test_mask, y, target,
                            train_donors, test_donors, args.device)
        m.update({"arm": arm, "n_train_donors": len(train_donors),
                  "n_test_donors": len(test_donors)})
        rows.append(m)
        if pred is not None:
            preds[arm] = pred
        print(f"  {arm}: f1={m['identity_f1_probe']:.3f} "
              f"rho={m['path_rho']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    for arm, pred in preds.items():
        pred.to_csv(out / f"predictions_{arm}.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "step": "v2_scfm_eval",
        "fold": {"path": str(args.fold.resolve()),
                 "sha256": file_sha256(args.fold)},
        "fold_manifest": str(args.fold_manifest),
        "embs": {Path(n).name: file_sha256(n)
                 for n in sorted(glob.glob(args.emb))},
        "label_col": args.label_col, "pathology_col": args.pathology_col,
        "protocol": "v2 frozen-embedding eval: linear_probe(200ep) + "
                    "donor-mean ridge, train donors -> test donors",
    }, indent=2))
    print(f"wrote {out} ({len(rows)} arms)")


if __name__ == "__main__":
    main()
