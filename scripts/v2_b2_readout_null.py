#!/usr/bin/env python3
"""B2 frozen-features readout-permutation null (review verification).

The existing B2 null (scripts/v2_perm_null.py) retrains the encoder on
permuted endpoint labels but fits the donor ridge on the TRUE target —
an endpoint-supervision null replicating N1. This script closes the
loop: it takes the REAL B2 checkpoint (trained on true labels), encodes
donors once, and permutes the ridge target over the same shared
donor->label maps used by the run nulls. Expected ~0 if the readout
machinery is sound.

One invocation handles one cohort+seed; encodes once, then applies all
requested perm indices (cheap ridge fits).
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
from biocellai import train as _train  # noqa: E402
from biocellai.progression import donor_embeddings  # noqa: E402
from biocellai.transfer import encode_cells, load_encoder  # noqa: E402
from biocellai.v2run import donor_target  # noqa: E402

from v2_perm_null import permuted_target, _ridge_rho  # noqa: E402


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
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--perms", type=int, default=100)
    p.add_argument("--pathology-col", required=True)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    fold = ad.read_h5ad(args.fold)
    man = json.loads((args.run / "manifest.json").read_text())
    seed = int(man["seed"])
    tc = _train_config(man, args.device)

    is_test = fold.obs["is_test"].to_numpy()
    donors = fold.obs["donor_id"].astype(str)
    train_donors = sorted(donors[~is_test].unique())
    test_donors = sorted(donors[is_test].unique())
    target = donor_target(fold.obs, args.pathology_col)

    enc = load_encoder(args.run / "ckpt_B2.pt", fold.n_vars, tc.hidden,
                       tc.embed_dim, device=args.device)
    feats = donor_embeddings(
        encode_cells(enc, fold.X, device=args.device), donors)

    rows = []
    for k in range(args.perms):
        perm_seed = seed * 100003 + k
        pt = permuted_target(target, train_donors, perm_seed)
        rows.append({"arm": "B2_readout_perm", "perm": k,
                     "perm_seed": perm_seed,
                     "rho": _ridge_rho(feats, pt, train_donors,
                                       test_donors, args.ridge_alpha),
                     "n_train_donors": len(train_donors),
                     "n_test_donors": len(test_donors)})

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = args.out / "b2_readout_perm.csv"
    df.to_csv(out_csv, index=False)
    print(f"{args.run.parent.name}/{args.run.name}: "
          f"mean={df.rho.mean():+.3f} q95={df.rho.quantile(.95):.3f} "
          f"-> {out_csv}")


if __name__ == "__main__":
    main()
