"""G2 — CellWhisperer CLIP-v1 zero-shot embeddings on the SEA-AD all-region split.

Uses the upstream epigen/CellWhisperer repo (cloned at
$ROOT/repos/cellwhisperer, pip-installed editable --no-deps) plus the public
checkpoint cellwhisperer_clip_v1.ckpt (transcriptome = Geneformer 12L-30M
backbone, text = BioBERT v1.1, CLIP joint space, 2048-d L2-normalized).

Environment-compat shims applied here (documented in experiments/m8_sweep):
  - TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (torch>=2.6 default breaks ckpt load)
  - PretrainedConfig._experts_implementation_internal = None
    (transformers>=4.4x MoE attr absent from upstream configs)
  - UCE stub package in venv site-packages (UCE arch unused)
  - cellwhisperer clone patched: UCE imports guarded, transcriptome_processor
    attribute re-set after ProcessorMixin init, get_embs replaced by direct
    mean-pool of last hidden state (geneformer==0.0.1 API gone)
  - resources/geneformer-12L-30M = config+random-init scaffold; ckpt
    state_dict overwrites all weights during load_from_checkpoint
  - resources/ensembl_gene_symbol_map.csv built from gc30M name->id dict
  - VERY_COMMON_GENES sanity check patched (HVG-filtered input lacks the
    upstream marker set)

Caveats vs upstream protocol: input h5ad is log-normalized HVG-2000 (not raw
counts) — Geneformer rank tokenization is rank-based so monotone transforms
preserve token order, but only ~2000 genes are available for ranking vs the
full transcriptome in the original model.

Eval (identical protocol as our arms, scripts/m7_progression.py):
  - linear probe on frozen embeddings -> cell-type macro-F1
  - mean-pool per donor -> ridge CPS_Global on train donors ->
    Spearman rho on held-out donors (3 seeds).

Usage (slurm): python scripts/g2_cellwhisperer.py --out experiments/g2_cellwhisperer
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from biocellai.data import donor_split
from biocellai.progression import (ADNC_ORDER, BRAAK_ORDER, CERAD_ORDER,
                               ordinal_map, regress_heldout,
                               trajectory_correlation)

ROOT = Path("/beegfs/a474r867/biocellai")
CKPT = ROOT / "data/cellwhisperer/cellwhisperer_clip_v1.ckpt"
DATA = ROOT / "data/seaad_allregions_s0.h5ad"
DONORS = ROOT / "data/seaad_donors_allregions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/g2_cellwhisperer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4096,
                    help="cells per tokenization chunk (bounds memory)")
    ap.add_argument("--max-cells", type=int, default=0,
                    help=">0: subsample cells (smoke test)")
    args = ap.parse_args()

    from transformers import PretrainedConfig
    PretrainedConfig._experts_implementation_internal = None
    import cellwhisperer.jointemb.geneformer_model as gfm
    from cellwhisperer.utils.model_io import load_cellwhisperer_model
    from cellwhisperer.utils.processing import adata_to_embeds

    print("loading cellwhisperer", CKPT, flush=True)
    pl_model, _tok, proc = load_cellwhisperer_model(str(CKPT))
    model = pl_model.model
    print("model on", next(model.parameters()).device, flush=True)

    print("loading", DATA, flush=True)
    adata = ad.read_h5ad(DATA)
    if args.max_cells:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(adata.n_obs, args.max_cells, replace=False))
        adata = adata[idx]

    # upstream sanity check expects full-transcriptome var; our h5ad is HVG-2000
    gfm.VERY_COMMON_GENES = set(adata.var_names[:50].astype(str))

    n = adata.n_obs
    emb = None
    for i in range(0, n, args.chunk):
        sub = adata[i:i + args.chunk].copy()
        e = adata_to_embeds(sub, model, proc, batch_size=args.batch)
        e = e.detach().cpu().numpy().astype(np.float32)
        emb = e if emb is None else np.vstack([emb, e])
        print(f"  {min(i + args.chunk, n)}/{n}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    donors = adata.obs["donor_id"].astype(str).to_numpy()
    ctypes = adata.obs["cell_type"].astype(str).to_numpy()
    np.savez_compressed(out / "cellwhisperer_emb.npz",
                        emb=emb, donor=donors, cell_type=ctypes,
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
        clf = LogisticRegression(max_iter=300, n_jobs=8)
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
    df.to_csv(out / "g2_cellwhisperer_metrics.csv", index=False)
    print(df.groupby("metric")["value"].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
