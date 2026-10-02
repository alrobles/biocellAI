#!/usr/bin/env python3
"""M12-S2 — fine-tune the scGPT whole-human backbone with the M10/M11
dual-head contrastive objective (ADR-009).

Mirrors `experiment m2` (same load_dataset path, donor_split seeds,
captions, eval npz schema) but the cell encoder is the pretrained
TransformerModel instead of the from-scratch MLP. Input = post-preprocess
cells restricted to in-vocab genes; official DataCollator tokenisation
(identical to G2 embed_data).

One (seed, text_model) pair trains one backbone — the objective is
encoder-specific. Array jobs pass a single seed/model per task.

Usage:
    python scripts/m12_scgpt_ft.py \
        --dataset $ROOT/data/seaad_allregions_s0.h5ad \
        --scgpt-dir $ROOT/data/scgpt_whole_human \
        --seeds 0 --text-models cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
        --type-descriptions data/text/m7/type_desc_pubmed_rag_markers_alias_allregions.json \
        --pathology-descriptions data/text/m11/pathology_cognitive_rag.json \
        --dual-head cognitive_status --ft-mode lastN --ft-blocks 2 \
        --lr 1e-4 --epochs 5 --batch-size 32 --max-cells-epoch 120000 \
        --outdir experiments/m12_s2_lastN --save-embeddings .../embeddings
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from biocellai.captions import build_cell_captions, build_class_captions, marker_table
from biocellai.data import donor_split
from biocellai.experiment import load_dataset
from biocellai.model import TextEncoder
from biocellai.scgpt_enc import (
    ScgptCellSet,
    emb_dim,
    encode_cells,
    gene_ids_in_vocab,
    load_scgpt,
    make_collator,
    train_contrastive_scgpt,
)
from biocellai.train import TrainConfig, encode_labels, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--scgpt-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--text-models", nargs="+", required=True)
    p.add_argument("--tissue", default="brain")
    p.add_argument("--type-descriptions", required=True)
    p.add_argument("--class-descriptions", default=None)
    p.add_argument("--label-key", default="cell_type")
    p.add_argument("--dual-head", required=True)
    p.add_argument("--pathology-descriptions", required=True)
    p.add_argument("--shuffle-donor-labels", action="store_true")
    p.add_argument("--ft-mode", choices=["full", "lastN"], default="lastN")
    p.add_argument("--ft-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=1200)
    p.add_argument("--max-cells-epoch", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--save-embeddings", default=None)
    p.add_argument("--tag", default="scgpt")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    type_desc = json.loads(Path(args.type_descriptions).read_text())
    class_desc = (json.loads(Path(args.class_descriptions).read_text())
                  if args.class_descriptions else None)
    path_desc = json.loads(Path(args.pathology_descriptions).read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adata, _ = load_dataset(args.dataset, seed=args.seeds[0],
                            max_cells=0, n_hvg=2000)
    if args.shuffle_donor_labels:
        rng0 = np.random.default_rng(args.seeds[0])
        dlab = adata.obs.groupby("donor_id", observed=True)[
            args.dual_head].first()
        donor2lab = dict(zip(dlab.index, rng0.permutation(dlab.values)))
        adata.obs[args.dual_head] = adata.obs["donor_id"].map(donor2lab)

    # model load once for vocab + gene matching (weights reloaded per run)
    model, vocab, mcfg = load_scgpt(args.scgpt_dir, device)
    gid = gene_ids_in_vocab(adata, vocab)
    keep = gid >= 0
    print(f"genes: {adata.n_vars} -> in vocab {keep.sum()}", flush=True)
    gid = gid[keep]
    Xsub = adata[:, keep].X
    cls_id, pad_id = vocab["<cls>"], vocab[mcfg["pad_token"]]
    collator = make_collator(vocab, mcfg, max_length=args.max_length)
    del model  # fresh per (seed, encoder) run below

    encoders = {n: TextEncoder(n, device=device) for n in args.text_models}
    y, cats = encode_labels(adata, key=args.label_key)
    rows, losses = [], {}
    cfg = TrainConfig(seed=0, epochs=args.epochs, device=device,
                      temperature=args.temperature)

    for seed in args.seeds:
        tr_mask, te_mask = donor_split(adata, seed=seed)
        tr = adata[tr_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        markers = marker_table(tr, n_markers=5)
        class_caps = build_class_captions(markers, tissue=args.tissue)
        caps_tr = build_cell_captions(
            tr, include_label=False, tissue=args.tissue, k_genes=8,
            mode="type_llm", markers=markers, type_descriptions=type_desc,
            label_key=args.label_key)
        pcaps_tr = [f"A cell from human {args.tissue}. "
                    f"{path_desc.get(str(l), 'No description.')}"
                    for l in tr.obs[args.dual_head]]

        for tm_name, tex in encoders.items():
            emb_t = tex.embed(caps_tr)
            emb_p = tex.embed(pcaps_tr)
            class_emb = tex.embed([class_caps[c] for c in cats])
            class_emb_m = (tex.embed([class_desc.get(c, class_caps[c])
                                      for c in cats]) if class_desc else None)

            model, _, _ = load_scgpt(args.scgpt_dir, device)
            train_set = ScgptCellSet(Xsub[tr_mask], gid, cls_id,
                                     mcfg["pad_value"])
            model, proj_t, proj_p, hist, meta = train_contrastive_scgpt(
                model, train_set, emb_t, emb_p, cfg, collator, device,
                pad_id=pad_id, ft_mode=args.ft_mode, ft_blocks=args.ft_blocks,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                max_cells_epoch=args.max_cells_epoch,
                donors_per_pos=tr.obs["donor_id"].astype(str).to_numpy(),
                seed=seed)
            tag = f"{args.tag}_{args.ft_mode}"
            losses[f"{tag}_s{seed}_{tm_name}"] = hist

            eval_set = ScgptCellSet(Xsub, gid, cls_id, mcfg["pad_value"])
            torch.manual_seed(seed)  # deterministic truncation at eval
            H512 = encode_cells(model, eval_set, collator, args.batch_size,
                                device, pad_id, emb_dim(model))
            Ht = torch.from_numpy(H512).to(device)
            with torch.no_grad():
                z_p = proj_p(Ht).cpu().numpy()
                z_t = proj_t(Ht).cpu().numpy()

            # cell-type metrics on the type head (same protocol as run_m2)
            zt = torch.from_numpy(z_t).to(device)
            E = F.normalize(zt[te_mask], dim=-1)
            sims = E @ F.normalize(class_emb.to(device), dim=-1).T
            rows.append(dict(seed=seed, arm="grounded_zeroshot",
                             text_model=tm_name,
                             **metrics(y_te, sims.argmax(-1).cpu().numpy())))
            if class_emb_m is not None:
                sims_m = E @ F.normalize(class_emb_m.to(device), dim=-1).T
                rows.append(dict(seed=seed, arm="grounded_zeroshot_matched",
                                 text_model=tm_name,
                                 **metrics(y_te, sims_m.argmax(-1).cpu().numpy())))
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=300)
            clf.fit(z_t[tr_mask], y_tr)
            rows.append(dict(seed=seed, arm="grounded_probe",
                             text_model=tm_name,
                             **metrics(y_te, clf.predict(z_t[te_mask]))))

            if args.save_embeddings:
                sd = Path(args.save_embeddings)
                sd.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    sd / f"emb_s{seed}_{tag}_{tm_name.split('/')[-1]}.npz",
                    emb=z_p, emb_type=z_t,
                    donor=adata.obs["donor_id"].astype(str).to_numpy(),
                    cell_type=adata.obs["cell_type"].astype(str).to_numpy(),
                    region=adata.obs["region"].astype(str).to_numpy()
                    if "region" in adata.obs.columns
                    else np.array([""] * adata.n_obs),
                    is_test=te_mask)
                torch.save(
                    dict(model=model.state_dict(), proj_t=proj_t.state_dict(),
                         proj_p=proj_p.state_dict(), vocab_size=len(vocab),
                         ft_mode=args.ft_mode, ft_blocks=args.ft_blocks,
                         genes_in_vocab=int(keep.sum())),
                    sd / f"model_s{seed}_{tag}_{tm_name.split('/')[-1]}.pt")

            prov = {**vars(args), "trainable_params": meta["trainable_params"],
                    "epoch_cells": meta["epoch_cells"], "seed": seed,
                    "text_model": tm_name,
                    "genes_in_vocab": int(keep.sum()), "n_obs": adata.n_obs}
            (outdir / f"prov_{tag}_s{seed}_{tm_name.split('/')[-1]}.json") \
                .write_text(json.dumps(prov, indent=2, default=str))
            del model
            torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)
    (outdir / "losses.json").write_text(json.dumps(losses))
    print(pd.DataFrame(rows)
          .groupby(["arm", "text_model"])[["macro_f1", "balanced_acc"]]
          .mean().round(4))


if __name__ == "__main__":
    main()
