#!/usr/bin/env python3
"""M12-S1 — extract scGPT embeddings aligned to the post-preprocess dataset.

The G2 scGPT embedding npz covers the raw s0 h5ad (all cells), while the
experiment pipeline re-runs preprocess() on load — a strict subset.
This maps the embeddings onto the processed cells via obs_names and
writes an npz with obs_names so `--input-emb` can re-verify alignment.

Usage:
    python scripts/m12_extract_scgpt_input.py \
        --h5ad /beegfs/a474r867/biocellai/data/seaad_allregions_s0.h5ad \
        --emb experiments/g2_scgpt/scgpt_emb.npz \
        --out /beegfs/a474r867/biocellai/data/scgpt_emb_seaad_s0proc.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from biocellai.experiment import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--emb", required=True, help="g2-style scgpt_emb.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    z = np.load(args.emb, allow_pickle=True)
    raw = ad.read_h5ad(args.h5ad, backed="r")
    assert len(z["donor"]) == raw.n_obs, "emb npz must be raw-h5ad aligned"

    proc, _ = load_dataset(args.h5ad, seed=args.seed, max_cells=0,
                           n_hvg=args.n_hvg)
    pos = raw.obs_names.astype(str).get_indexer(proc.obs_names.astype(str))
    assert (pos >= 0).all() and len(np.unique(pos)) == len(pos)

    emb = np.asarray(z["emb"], dtype=np.float32)[pos]
    for key, col in (("donor", "donor_id"), ("cell_type", "cell_type")):
        if key in z.files:
            assert np.array_equal(
                z[key].astype(str)[pos],
                proc.obs[col].astype(str).to_numpy()), f"{key} mismatch"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, emb=emb, obs_names=proc.obs_names.astype(str).to_numpy(),
        donor=proc.obs["donor_id"].astype(str).to_numpy(),
        cell_type=proc.obs["cell_type"].astype(str).to_numpy(),
        region=proc.obs["region"].astype(str).to_numpy()
        if "region" in proc.obs.columns else np.array([""] * proc.n_obs))
    print(f"wrote {out}: emb {emb.shape}, cells {proc.n_obs}")


if __name__ == "__main__":
    main()
