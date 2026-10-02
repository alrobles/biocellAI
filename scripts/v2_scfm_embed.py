"""Embed v2 fold cells with an external scFM checkpoint (R4-BENCHMARK).

Produces <out>.npz (emb, obs_names, donor, cell_type) plus
<out>.manifest.json recording model revision, vocab coverage, input
mode, timing, device, and every environment patch applied.

    python scripts/v2_scfm_embed.py \
        --extract data/v2_src/rosmap_liu2025.h5ad \
        --fold experiments/v2_revalidation/rosmap/folds/fold_s0.h5ad \
        --model gf_v2_104m --input native \
        --out experiments/v2_revalidation/rosmap/scfm/gf_v2_104m_native_s0
"""
from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from biocellai.revalidation import file_sha256
from biocellai.scfm import MODELS, embed, load_fold_cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract", type=Path, required=True,
                   help="v2_src extract h5ad (full-gene counts)")
    p.add_argument("--fold", type=Path, required=True,
                   help="prepared fold h5ad; defines the cell set/order")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--input", choices=["native", "hvg"], default="native")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gf-snapshot", type=Path, default=None)
    p.add_argument("--scgpt-dir", type=Path, default=None)
    p.add_argument("--cw-ckpt", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    adata = load_fold_cells(args.extract, args.fold, args.input)
    paths = {"gf_snapshot": args.gf_snapshot,
             "scgpt_dir": args.scgpt_dir, "cw_ckpt": args.cw_ckpt}
    emb, keep_idx, info = embed(adata, args.model, paths,
                                batch_size=args.batch_size,
                                device=args.device)

    npz_path = args.out.with_suffix(".npz")
    man_path = args.out.with_suffix(".manifest.json")
    for f in (npz_path, man_path):
        if f.exists():
            raise FileExistsError(f"refusing to overwrite {f}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    obs_names = np.array([str(x) for x in adata.obs_names[keep_idx]])
    np.savez_compressed(
        npz_path, emb=emb.astype(np.float32),
        obs_names=obs_names,
        donor=np.array([str(x) for x in
                        adata.obs["donor_id"].to_numpy()[keep_idx]]),
        cell_type=np.array([str(x) for x in
                            adata.obs["cell_type"].to_numpy()[keep_idx]]),
        keep_idx=keep_idx.astype(np.int64))
    manifest = {
        "step": "v2_scfm_embed",
        "model": args.model, "input_mode": args.input,
        "extract": {"path": str(args.extract.resolve()),
                    "sha256": file_sha256(args.extract)},
        "fold": {"path": str(args.fold.resolve()),
                 "sha256": file_sha256(args.fold)},
        "n_cells": int(adata.n_obs), "n_cells_embedded": int(len(keep_idx)),
        "n_genes_input": int(adata.n_vars),
        "emb_dim": int(emb.shape[1]),
        "device": args.device or "auto",
        "model_info": info,
        "total_wall_s": round(time.time() - t0, 1),
        "environment": {n: version(n)
                        for n in ("numpy", "torch", "transformers",
                                  "anndata", "scipy")},
    }
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {npz_path} emb={emb.shape} genes_in={adata.n_vars} "
          f"wall={manifest['total_wall_s']}s")


if __name__ == "__main__":
    main()
