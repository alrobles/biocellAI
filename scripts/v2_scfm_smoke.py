"""R4-SCGPT-CW smoke tests: frozen embedding sanity + fine-tune connectivity.

For each model: load the audited checkpoint, embed a small cell subset
(frozen), then run a few gradient steps proving the train graph is
connected (loss finite, backbone parameters actually receive gradients
and update). Output JSON becomes part of the audit evidence.

    python scripts/v2_scfm_smoke.py --model scgpt_wh \
        --extract data/v2_src/seaad_mtg.h5ad \
        --fold experiments/v2_revalidation/seaad_mtg/folds/fold_s0.h5ad \
        --scgpt-dir ... --cw-ckpt ... --n-cells 256 \
        --out experiments/v2_revalidation/scfm_audit/smoke_scgpt_wh.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from biocellai.scfm import load_fold_cells


def _take(adata, n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=min(n, adata.n_obs), replace=False)
    return adata[np.sort(idx)].copy()


def smoke_scgpt(adata, model_dir, device="cpu", n_steps=10):
    from biocellai.scgpt_enc import (ScgptCellSet, emb_dim,
                                   gene_ids_in_vocab, load_scgpt,
                                   make_collator)
    from torch.utils.data import DataLoader, SequentialSampler

    model, vocab, cfg = load_scgpt(model_dir, device)
    gid = gene_ids_in_vocab(adata, vocab)
    keep = gid >= 0
    Xsub = adata[:, keep].X
    gid = gid[keep]
    collator = make_collator(vocab, cfg, max_length=600)
    pad_id = vocab[cfg["pad_token"]]
    ds = ScgptCellSet(Xsub, gid, vocab["<cls>"], cfg["pad_value"])
    loader = DataLoader(ds, batch_size=32, sampler=SequentialSampler(ds),
                        collate_fn=collator)
    model.train()
    head = torch.nn.Linear(emb_dim(model), 1).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(list(head.parameters()) + params[-200:],
                            lr=1e-4)
    tgt = torch.randn(len(ds), 1)
    before = params[-1].detach().clone()
    losses, batch = [], next(iter(loader))
    for step in range(n_steps):
        g = batch["gene"].to(device)
        m = g.eq(pad_id)
        h = model._encode(g, batch["expr"].to(device),
                          src_key_padding_mask=m)[:, 0, :]
        loss = torch.nn.functional.mse_loss(
            head(h), tgt[:len(h)].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    grads = sum(p.grad.abs().sum().item() for p in params
                if p.grad is not None)
    return {"losses": losses, "loss_first": losses[0],
            "loss_last": losses[-1],
            "backbone_grad_sum": grads,
            "param_changed": bool((params[-1] - before).abs().sum() > 0)}


def smoke_cw(adata, ckpt_path, device="cpu"):
    import os
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from transformers import PretrainedConfig
    PretrainedConfig._experts_implementation_internal = None
    from cellwhisperer.utils.model_io import load_cellwhisperer_model
    from cellwhisperer.utils.processing import adata_to_embeds
    import cellwhisperer.jointemb.geneformer_model as gfm

    pl_model, _tok, proc = load_cellwhisperer_model(str(ckpt_path))
    model = pl_model.model
    adata = adata.copy()
    if "ensembl_id" not in adata.var.columns and "feature_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["feature_id"].astype(str)
    if "gene_name" not in adata.var.columns:
        src = "feature_name" if "feature_name" in adata.var.columns else None
        adata.var["gene_name"] = (adata.var[src].astype(str) if src
                                  else adata.var_names.astype(str))
    gfm.VERY_COMMON_GENES = set(adata.var["gene_name"].head(200))
    model.to(device)
    # frozen smoke first: detached embeds through the shipped wrapper
    sub = adata[: min(64, adata.n_obs)].copy()
    emb_frozen = adata_to_embeds(sub, model, proc, batch_size=16)
    # FT connectivity: released ckpt wraps the transcriptome tower in
    # FrozenCachedModel (locking_mode=LL -> cached detached features).
    # Unwrap the inner tower, unfreeze, forward the same tokens with
    # grad enabled, backward a scalar loss, verify backbone grads.
    tower = model.transcriptome_model.model
    tower = tower.to(device).train()
    tower.requires_grad_(True)
    tok = proc(sub, return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in tok.items()}
    out = tower(**batch)
    # CW GeneformerModel.forward returns (None, embs): pooled cell emb
    feat = out[-1] if isinstance(out, (tuple, list)) else out
    if feat.dim() == 3:
        feat = feat.mean(dim=1)
    loss = feat.float().square().mean()
    loss.backward()
    tparams = [p for n, p in tower.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    return {"fwd_dim": list(emb_frozen.shape),
            "loss": float(loss.item()),
            "tower": type(tower).__name__,
            "locking_mode": str(model.config.locking_mode),
            "n_params_with_grad": len(tparams)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["scgpt_wh", "cw_clip_v1"])
    p.add_argument("--extract", type=Path, required=True)
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--scgpt-dir", type=Path)
    p.add_argument("--cw-ckpt", type=Path)
    p.add_argument("--n-cells", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    adata = _take(load_fold_cells(args.extract, args.fold, "native"),
                  args.n_cells)
    rec = {"step": "v2_scfm_smoke", "model": args.model,
           "n_cells": int(adata.n_obs), "device": args.device}
    t0 = time.time()
    if args.model == "scgpt_wh":
        rec.update(smoke_scgpt(adata, args.scgpt_dir, args.device))
    else:
        rec.update(smoke_cw(adata, args.cw_ckpt, args.device))
    rec["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
