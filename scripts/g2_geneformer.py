"""G2 — Geneformer zero-shot embeddings on the SEA-AD all-region split.

Self-contained: uses the HF snapshot of ctheodoris/Geneformer (which
bundles token dictionaries + Ensembl<->symbol map + V1/V2 weights)
so it runs under HF_HUB_OFFLINE=1 with no geneformer package install.

Tokenization (Geneformer V2): per cell, normalize counts per gene by
the Genecorpus-30M gene median, rank descending -> sequence of Ensembl
token ids (max_len 2048). Cell embedding = mean of last hidden state
(standard Geneformer cell embedding).

Eval (identical protocol as our arms, scripts/m7_progression.py):
  - linear probe on frozen embeddings -> cell-type macro-F1
  - mean-pool per donor -> ridge CPS_Global on train donors ->
    Spearman rho on held-out donors (3 seeds).

Usage (slurm): python scripts/g2_geneformer.py --model Geneformer-V1-10M
"""
from __future__ import annotations

import argparse
import pickle
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
SNAP = (ROOT / "hf_cache/hub/models--ctheodoris--Geneformer/snapshots"
        "/1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5")
DATA = ROOT / "data/seaad_allregions_s0.h5ad"
DONORS = ROOT / "data/seaad_donors_allregions.csv"
MAX_LEN = 2048


def rank_tokenize(counts_row, gene_tokens, gene_medians, max_len=MAX_LEN):
    """counts_row: dense np.array aligned to gene_tokens order."""
    norm = counts_row / gene_medians
    order = np.argsort(-norm)
    nz = norm[order] > 0
    return gene_tokens[order][nz][:max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Geneformer-V1-10M")
    ap.add_argument("--out", default="experiments/g2_geneformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-cells", type=int, default=0,
                    help=">0: subsample cells (smoke test)")
    args = ap.parse_args()

    pkg = SNAP / "geneformer"
    v2 = "V2" in args.model
    tok_pkl = (pkg / "token_dictionary_gc104M.pkl") if v2 else \
        (pkg / "gene_dictionaries_30m/token_dictionary_gc30M.pkl")
    med_pkl = (pkg / "gene_median_dictionary_gc104M.pkl") if v2 else \
        (pkg / "gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl")
    map_pkl = (pkg / "gene_name_id_dict_gc104M.pkl") if v2 else \
        (pkg / "gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl")
    with open(tok_pkl, "rb") as f:
        tok_dict = pickle.load(f)            # {ensembl_id: token}
    with open(med_pkl, "rb") as f:
        med_dict = pickle.load(f)            # {ensembl_id: median}
    with open(map_pkl, "rb") as f:
        name2ens = pickle.load(f)            # {symbol: ensembl_id}

    print("loading", DATA, flush=True)
    adata = ad.read_h5ad(DATA)
    X = adata.X.tocsr() if hasattr(adata.X, "tocsr") else adata.X
    symbols = np.asarray(adata.var_names.astype(str))

    ens = np.array([name2ens.get(s) for s in symbols], dtype=object)
    keep = np.array([e in tok_dict and e in med_dict for e in ens])
    print(f"genes: {len(symbols)} -> mapped {keep.sum()}", flush=True)
    gene_tokens = np.array([tok_dict[e] for e in ens[keep]])
    gene_med = np.array([med_dict[e] for e in ens[keep]], dtype=np.float64)
    X = X[:, keep]

    if args.max_cells:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(adata.n_obs, args.max_cells, replace=False))
        adata = adata[idx]
        X = X[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = str(SNAP / args.model)
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained(model_dir).bert.to(device).eval()
    print("model", args.model, "on", device, flush=True)

    n = adata.n_obs
    emb = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    for i in range(0, n, args.batch):
        Xi = X[i:i + args.batch].toarray()
        seqs = [rank_tokenize(r, gene_tokens, gene_med) for r in Xi]
        L = min(max(len(s) for s in seqs), MAX_LEN)
        ids = np.zeros((len(seqs), L), dtype=np.int64)
        mask = np.zeros((len(seqs), L), dtype=np.int64)
        pad = tok_dict.get("<pad>", 0)
        for j, s in enumerate(seqs):
            Ls = min(len(s), L)
            ids[j, :Ls] = s[:Ls]
            mask[j, :Ls] = 1
            ids[j, Ls:] = pad
        with torch.no_grad():
            out = model(input_ids=torch.from_numpy(ids).to(device),
                        attention_mask=torch.from_numpy(mask).to(device))
        h = out.last_hidden_state
        m = torch.from_numpy(mask).to(device).unsqueeze(-1).float()
        emb[i:i + len(seqs)] = ((h * m).sum(1) / m.sum(1).clamp(min=1)
                                ).cpu().numpy()
        if i % (args.batch * 100) == 0:
            print(f"  {i}/{n}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    donors = adata.obs["donor_id"].astype(str).to_numpy()
    ctypes = adata.obs["cell_type"].astype(str).to_numpy()
    np.savez_compressed(out / f"gf_{args.model}_emb.npz",
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
        # cell-type linear probe (frozen GF embedding)
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
        # donor-level CPS regression
        r = regress_heldout(demb, cps, tr_d, te_d)
        rows.append(dict(seed=seed, metric="donor_CPS_Global_rho",
                         value=r["rho"]))
        rows.append(dict(seed=seed, metric="donor_CPS_Global_r2",
                         value=r["r2"]))
        # ordinal trajectories (PC1 oriented on train donors, eval on test)
        for key, t in ordinals.items():
            tr = trajectory_correlation(demb, t, set(tr_d))
            rows.append(dict(seed=seed, metric=f"{key}_trajectory_rho",
                             value=tr["rho"]))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"g2_{args.model}_metrics.csv", index=False)
    print(df.groupby("metric")["value"].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
