#!/usr/bin/env python3
"""M11-A2 — project ROSMAP cells onto the SEA-AD 2000-HVG input space.

Builds rosmap_for_seaad.h5ad: the same cells as rosmap_s0.h5ad (same donor
sampling, obs names preserved) but with X = raw ROSMAP counts restricted to
SEA-AD's 2000 HVGs (all present in the ROSMAP gene space), then
normalize_total(1e4) + log1p — identical preprocessing to the SEA-AD s0
pipeline so a SEA-AD-trained encoder can consume it directly.

This single aligned matrix serves both transfer directions:
  - SEA-AD->ROSMAP: apply the SEA-AD encoder to this file
  - ROSMAP->SEA-AD: train a ROSMAP pathology model on this file, then apply
    it to seaad_allregions_s0.h5ad (identical input space, zero padding)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw ROSMAP h5ad (36k genes)")
    ap.add_argument("--s0", required=True, help="rosmap_s0.h5ad (cell subset)")
    ap.add_argument("--genes-from", required=True,
                    help="h5ad whose var_names define the target gene space")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = ad.read_h5ad(args.genes_from, backed="r")
    target_genes = list(map(str, ref.var_names))
    del ref

    s0 = ad.read_h5ad(args.s0, backed="r")
    keep_names = list(s0.obs_names)
    obs = s0.obs.copy()
    del s0
    print(f"target cells: {len(keep_names)}; target genes: {len(target_genes)}",
          flush=True)

    raw = ad.read_h5ad(args.raw, backed="r")
    raw_names = raw.obs_names
    pos = raw_names.get_indexer(keep_names)
    assert (pos >= 0).all(), "s0 cells missing from raw"
    gpos = raw.var_names.get_indexer(target_genes)
    n_missing = int((gpos < 0).sum())
    print(f"genes missing from raw: {n_missing}", flush=True)
    present = [i for i, g in enumerate(gpos) if g >= 0]
    target_keep = [target_genes[i] for i in present]
    raw_gpos = gpos[present]

    # materialise cells x present genes (columns follow sorted(raw_gpos)),
    # then reorder to the requested gene order by name
    sub = raw[pos, sorted(raw_gpos)].to_memory()
    sub = sub[:, target_keep].copy()
    sub.obs = obs.copy()

    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)

    # pad any missing genes with zeros, in exact target order
    if n_missing:
        full = ad.AnnData(
            X=np.zeros((sub.n_obs, len(target_genes)), dtype=np.float32),
            obs=sub.obs)
        col = {g: i for i, g in enumerate(target_genes)}
        for j, g in enumerate(sub.var_names):
            full.X[:, col[g]] = np.asarray(
                sub.X[:, j].todense() if hasattr(sub.X[:, j], "todense")
                else sub.X[:, j]).ravel()
        full.var_names = target_genes
        sub = full
    else:
        sub.var_names = target_genes

    sub.write_h5ad(args.out)
    manifest = dict(n_cells=int(sub.n_obs), n_genes=int(sub.n_vars),
                    genes_missing=n_missing,
                    source_raw=str(args.raw), gene_space=str(args.genes_from))
    Path(str(args.out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
