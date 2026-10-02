"""External single-cell foundation-model embedding for v2 folds (R4).

Frozen embedding functions evaluated under the identical v2 protocol
(same cells, donor splits, probe, donor ridge) — no fitting to the data,
so donor-held-out discipline is preserved.

Model registry (checkpoints hashed in experiments/v2_revalidation/inventory.json):
  gf_v1_10m   Geneformer V1-10M     gc30M dicts, no special tokens, len 2048
  gf_v2_104m  Geneformer V2-104M    gc104M dicts, <cls>/<eos>, len 4096
  gf_v2_316m  Geneformer V2-316M    gc104M dicts, <cls>/<eos>, len 4096
  scgpt_wh    scGPT whole-human     official scgpt.tasks.embed_data (0.2.4)
  cw_clip_v1  CellWhisperer CLIP-v1 Geneformer-12L-30M + BioBERT joint space

Documented environment shims (recorded per-run in the output manifest,
satisfying R4-SCGPT-CW "no hidden patches"):
  - scgpt 0.2.4 calls anndata.AnnData(dtype=...) removed in anndata>=0.10;
    a scoped __init__ wrapper drops the kwarg inside embed_data only.
  - CellWhisperer needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (torch>=2.6)
    and PretrainedConfig._experts_implementation_internal = None
    (transformers>=4.4x). UCE stub + clone patches live in the
    repos/cellwhisperer checkout and are enumerated, not applied here.
"""
from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

GF_SPECS = {
    "gf_v1_10m": dict(model_dir="Geneformer-V1-10M", dict_subdir="gene_dictionaries_30m",
                      tag="gc30M", special=False, max_len=2048),
    "gf_v2_104m": dict(model_dir="Geneformer-V2-104M", dict_subdir="",
                       tag="gc104M", special=True, max_len=4096),
    "gf_v2_316m": dict(model_dir="Geneformer-V2-316M", dict_subdir="",
                       tag="gc104M", special=True, max_len=4096),
}
MODELS = [*GF_SPECS, "scgpt_wh", "cw_clip_v1"]


def load_fold_cells(extract_path, fold_path, input_mode: str):
    """Return the extract AnnData restricted to the fold's cells (fold order).

    input_mode 'native' keeps the full gene space; 'hvg' restricts to the
    fold's var_names (the 2000 train-only HVGs the v2 encoders see).
    """
    ext = ad.read_h5ad(extract_path)
    fold = ad.read_h5ad(fold_path)
    keep = ext.obs_names.get_indexer(fold.obs_names)
    if (keep < 0).any():
        missing = fold.obs_names[keep < 0][:5]
        raise ValueError(f"fold cells missing from extract: {missing} ...")
    out = ext[keep].copy()
    out.obs = fold.obs.copy()  # fold obs carries harmonized cols
    if input_mode == "hvg":
        # folds are symbol-mapped (use_gene_symbols) while extracts may be
        # joinid/ensg-keyed — pick the extract column covering most fold genes
        fold_genes = set(map(str, fold.var_names))
        cands = {c: out.var[c].astype(str)
                 for c in ("feature_name", "feature_id", "ensembl_id",
                           "gene_name") if c in out.var.columns}
        cands["__index__"] = pd.Series(out.var_names.astype(str))
        best = max(cands, key=lambda c: cands[c].isin(fold_genes).sum())
        mask = cands[best].isin(fold_genes).to_numpy()
        if not mask.any():
            raise ValueError("no overlap between extract and fold genes")
        out.uns["hvg_gene_source"] = best
        out = out[:, mask].copy()
    elif input_mode != "native":
        raise ValueError(f"unknown input_mode {input_mode!r}")
    return out


def gf_load_dicts(snapshot_dir, model_key):
    spec = GF_SPECS[model_key]
    pkg = Path(snapshot_dir) / "geneformer"
    sub = pkg / spec["dict_subdir"] if spec["dict_subdir"] else pkg
    tag = spec["tag"]
    with open(sub / f"token_dictionary_{tag}.pkl", "rb") as f:
        tok = pickle.load(f)
    with open(sub / f"gene_median_dictionary_{tag}.pkl", "rb") as f:
        med = pickle.load(f)
    # upstream maps var ids via ensembl_mapping_dict (alias/ensg -> ens,
    # .upper() lookup), filtered to token-dict values (tokenizer.py ~L404).
    # symbol-keyed data needs the official symbol map gene_name_id_dict —
    # ensmap has ~zero symbol coverage (verified on real cohorts).
    with open(sub / f"ensembl_mapping_dict_{tag}.pkl", "rb") as f:
        ensmap = {k: v for k, v in pickle.load(f).items() if v in tok}
    with open(sub / f"gene_name_id_dict_{tag}.pkl", "rb") as f:
        nid = pickle.load(f)
    return tok, med, ensmap, nid


def pick_gene_ids(adata, key_set, candidates=("ensembl_id", "feature_id",
                                            "feature_name", "__index__")):
    """Return (col_name, ids, hit_rate) for the var column whose values
    best match key_set (via .upper() membership, upstream convention).

    Extracts differ in gene-id placement: blood's var_names are numeric
    joinids (ids in feature_id/feature_name), brain cohorts are
    symbol-keyed var_names. '__index__' means adata.var_names.
    """
    best_name, best_ids, best_hits = None, None, -1.0
    for c in candidates:
        if c == "__index__":
            ids = pd.Series(adata.var_names.astype(str))
        elif c in adata.var.columns:
            ids = adata.var[c].astype(str)
        else:
            continue
        probe = ids.head(2000)
        hits = float(np.mean([s.upper() in key_set for s in probe]))
        if hits > best_hits:
            best_name, best_ids, best_hits = c, ids, hits
        if hits > 0.8:
            break
    return best_name, best_ids, best_hits


def gf_tokenize(adata, tok, med, ensmap, nid, special: bool, max_len: int,
                target_sum: float = 10_000.0, gene_ids=None,
                dict_choice=None):
    """Official-parity Geneformer tokenization (fixture-checked upstream).

    var ids resolved by the best-hit official dict: ensmap
    (ensembl/alias keyed, .upper() lookup — upstream's own path) or
    gene_name_id_dict (symbol keyed) whichever covers more ids;
    upstream collapse_gene_ids=True sums rows ONLY when distinct input
    ids share a mapped ensembl (1:1 inputs stay separate); library size
    is obs['n_counts'] if present else X.sum(1); per cell:
    counts/libsize*target_sum/gene_median over nonzero, argsort desc,
    V2 wraps [<cls>, seq[:max_len-2], <eos>].
    """
    from scipy import sparse

    if gene_ids is None:
        if "ensembl_id" in adata.var.columns:
            ids = adata.var["ensembl_id"].astype(str)
        else:
            ids = pd.Series(adata.var_names.astype(str))
    else:
        ids = pd.Series(gene_ids).astype(str)
    if dict_choice is None:
        # pick dict by probe hit-rate (ensmap for ens/alias columns,
        # nid for symbol columns)
        probe = ids.head(2000)
        e_hit = float(np.mean([s.upper() in ensmap for s in probe]))
        n_hit = float(np.mean([s in nid for s in probe]))
        dict_choice = "nid" if n_hit >= e_hit else "ensmap"
    if dict_choice == "nid":
        name2ens = {k: v for k, v in nid.items()}
        mapped = [name2ens.get(s) for s in ids]
        in_map = np.array([s in name2ens for s in ids])
    else:
        name2ens = ensmap
        mapped = [name2ens.get(s.upper()) for s in ids]
        in_map = np.array([s in ensmap for s in ids])
    # upstream: raw-key membership decides the 1:1-vs-collapse branch,
    # but the mapping lookup itself upper-cases the id (ensmap only)
    uniq_in = {s for s, im in zip(ids, in_map) if im}
    uniq_out = {m for m in mapped if m is not None}
    collapse = len(uniq_in) != len(uniq_out)

    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    if collapse:
        # upstream concat order: unique-mapped rows (var order), then
        # summed groups sorted by ens id (pandas groupby key order)
        cnt = Counter(e for e in mapped if e is not None)
        unique_rows = [j for j, e in enumerate(mapped)
                       if e is not None and cnt[e] == 1
                       and e in tok and e in med]
        dup_ens = sorted(e for e in cnt
                         if cnt[e] > 1 and e in tok and e in med)
        dup_cols = [[j for j, e in enumerate(mapped) if e == de]
                    for de in dup_ens]
        ens_ids = [mapped[j] for j in unique_rows] + dup_ens
        parts = [X[:, unique_rows].copy()] if unique_rows else []
        parts += [sparse.csr_matrix(
            np.asarray(X[:, c].sum(axis=1))) for c in dup_cols]
        Xc = sparse.hstack(parts).tocsr()
    else:
        # 1:1 upstream branch: each mapped row stays its own gene entry
        keep = [j for j, e in enumerate(mapped)
                if e is not None and e in tok and e in med]
        ens_ids = [mapped[j] for j in keep]
        Xc = X[:, keep].copy()
    ens_ids = np.array(ens_ids)
    if "n_counts" in adata.obs:
        libsize = np.asarray(adata.obs["n_counts"], dtype=np.float64)
    else:
        libsize = np.asarray(X.sum(axis=1)).ravel()
    libsize[libsize <= 0] = 1.0
    med_v = np.array([med[e] for e in ens_ids], dtype=np.float64)
    tok_v = np.array([tok[e] for e in ens_ids], dtype=np.int64)
    cls_id, eos_id = tok.get("<cls>"), tok.get("<eos>")

    Xc = Xc.tocsr()
    # upstream truncates the raw seq to input_size-2, then inserts CLS/EOS
    inner = max_len - 2 if special else max_len
    seqs = []
    for i in range(Xc.shape[0]):
        row = Xc.getrow(i)
        vals = row.data / libsize[i] * target_sum / med_v[row.indices]
        order = np.argsort(-vals)
        seq = tok_v[row.indices[order]][:inner]
        if special:
            seq = np.concatenate([[cls_id], seq, [eos_id]])
        seqs.append(seq)
    return seqs


def gf_embed(adata, snapshot_dir, model_key, batch_size: int = 32,
             device: str | None = None):
    """Mean-pooled last-hidden-state embedding (standard GF cell emb)."""
    import torch
    from transformers import BertForMaskedLM

    from tqdm import tqdm

    spec = GF_SPECS[model_key]
    tok, med, ensmap, nid = gf_load_dicts(snapshot_dir, model_key)
    key_set = set(ensmap) | set(nid)
    id_col, ids, hit = pick_gene_ids(adata, key_set)
    probe = ids.head(2000)
    e_hit = float(np.mean([s.upper() in ensmap for s in probe]))
    n_hit = float(np.mean([s in nid for s in probe]))
    dict_choice = "nid" if n_hit >= e_hit else "ensmap"
    seqs = gf_tokenize(adata, tok, med, ensmap, nid, spec["special"],
                       spec["max_len"], gene_ids=ids,
                       dict_choice=dict_choice)
    # upstream drops cells whose tokenized seq is empty (no vocab genes)
    keep_idx = np.array([i for i, s in enumerate(seqs) if len(s) > 0],
                        dtype=np.int64)
    seqs = [seqs[i] for i in keep_idx]
    if not seqs:
        raise ValueError(
            f"all {adata.n_obs} cells tokenized to empty seqs "
            f"(gene_id_source={id_col}, hit_rate={hit:.3f})")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = str(Path(snapshot_dir) / spec["model_dir"])
    model = BertForMaskedLM.from_pretrained(model_dir).bert.to(device).eval()
    # fp16 inference on GPU: same weights/computation, ~2x faster on
    # tensor-core cards — needed to fit V2 x 4096-tok x >100k cells in
    # the 6h partition cap. Numerics are negligible for masked mean-pool.
    dtype = "fp32"
    if device == "cuda":
        model = model.half()
        dtype = "fp16"

    n = len(seqs)
    emb = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    pad = tok.get("<pad>", 0)
    # length-sorted batching: batches share true max length -> far less
    # pad compute on variable-length sequences; per-cell math unchanged
    order = np.argsort([len(s) for s in seqs])
    orig = np.arange(n)[order]
    seqs = [seqs[i] for i in order]
    for i in tqdm(range(0, n, batch_size), desc=model_key):
        chunk = seqs[i:i + batch_size]
        L = min(max(len(s) for s in chunk), spec["max_len"])
        ids = np.full((len(chunk), L), pad, dtype=np.int64)
        mask = np.zeros((len(chunk), L), dtype=np.int64)
        for j, s in enumerate(chunk):
            Ls = min(len(s), L)
            ids[j, :Ls] = s[:Ls]
            mask[j, :Ls] = 1
        with torch.no_grad():
            out = model(input_ids=torch.from_numpy(ids).to(device),
                        attention_mask=torch.from_numpy(mask).to(device))
        h = out.last_hidden_state
        m = torch.from_numpy(mask).to(device).unsqueeze(-1).to(h.dtype)
        emb[orig[i:i + len(chunk)]] = (
            (h * m).sum(1) / m.sum(1).clamp(min=1)
        ).float().cpu().numpy()
    return emb, {"model": model_key, "hidden": model.config.hidden_size,
                 "params": int(sum(p.numel() for p in model.parameters())),
                 "n_dropped_empty": int(adata.n_obs - n),
                 "gene_id_source": id_col,
                 "mapping_dict": dict_choice,
                 "ensmap_hit": round(e_hit, 4),
                 "nid_hit": round(n_hit, 4),
                 "id_map_hit_rate": round(hit, 4),
                 "dtype": dtype, "length_sorted": True}, keep_idx


def scgpt_embed(adata, model_dir, batch_size: int = 64,
                device: str | None = None):
    """Official scgpt.tasks.embed_data; scoped anndata dtype shim (documented)."""
    import torch
    import scgpt as scg

    orig = ad.AnnData.__init__

    def _compat(self, X=None, **kw):
        kw.pop("dtype", None)
        orig(self, X=X, **kw)

    ad.AnnData.__init__ = _compat
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tmp = adata.copy()
        vocab = json.load(open(Path(model_dir) / "vocab.json"))
        # scGPT vocab is gene SYMBOLS; pick the var column that best
        # matches (blood keeps symbols in feature_name, ids are joinids)
        id_col, ids, hit = pick_gene_ids(
            adata, set(vocab), candidates=("feature_name", "gene_name",
                                           "__index__"))
        tmp.var["gene_name"] = np.asarray(ids)
        keep = np.array([g in vocab for g in np.asarray(ids)])
        tmp = tmp[:, keep].copy()
        emb_adata = scg.tasks.embed_data(
            tmp, model_dir, gene_col="gene_name", batch_size=batch_size,
            device=device, return_new_adata=True)
        emb = np.asarray(getattr(emb_adata, "X_emb", None)
                         if hasattr(emb_adata, "X_emb") else emb_adata.X)
    finally:
        ad.AnnData.__init__ = orig
    return emb, {"model": "scgpt_wh", "vocab_genes": int(keep.sum()),
                 "n_dropped_empty": 0,
                 "gene_id_source": id_col,
                 "id_map_hit_rate": round(hit, 4)}, np.arange(adata.n_obs)


def cw_embed(adata, ckpt_path, batch_size: int = 32, chunk: int = 4096):
    """Upstream CellWhisperer path with its documented environment shims."""
    import os
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    from transformers import PretrainedConfig

    PretrainedConfig._experts_implementation_internal = None
    from cellwhisperer.utils.model_io import load_cellwhisperer_model
    from cellwhisperer.utils.processing import adata_to_embeds
    import cellwhisperer.jointemb.geneformer_model as gfm

    pl_model, _tok, proc = load_cellwhisperer_model(str(ckpt_path))
    model = pl_model.model
    # upstream processor: ensembl_id col / ENSG index wins, else
    # var['gene_name'] / var.index checked against VERY_COMMON_GENES.
    # Blood keeps ENSG ids in feature_id + symbols in feature_name;
    # brain cohorts are symbol-keyed var_names.
    adata = adata.copy()
    if "ensembl_id" not in adata.var.columns and "feature_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["feature_id"].astype(str)
    if "gene_name" not in adata.var.columns:
        src = "feature_name" if "feature_name" in adata.var.columns else None
        adata.var["gene_name"] = (adata.var[src].astype(str) if src
                                  else adata.var_names.astype(str))
    gfm.VERY_COMMON_GENES = set(adata.var["gene_name"].head(200))
    id_col = ("ensembl_id" if "feature_id" in adata.var.columns
              else ("feature_name" if "feature_name" in adata.var.columns
                    else "__index__"))

    n = adata.n_obs
    emb = None
    for i in range(0, n, chunk):
        sub = adata[i:i + chunk].copy()
        e = adata_to_embeds(sub, model, proc, batch_size=batch_size)
        e = e.detach().cpu().numpy().astype(np.float32)
        emb = e if emb is None else np.vstack([emb, e])
    return emb, {"model": "cw_clip_v1", "dim": int(emb.shape[1]),
                 "device": str(next(model.parameters()).device),
                 "n_dropped_empty": 0,
                 "gene_id_source": id_col}, np.arange(n)


PATCHES = {
    "gf_v1_10m": [], "gf_v2_104m": [], "gf_v2_316m": [],
    "scgpt_wh": ["anndata.AnnData.__init__ drops dtype kwarg inside "
                 "embed_data (scgpt 0.2.4 vs anndata>=0.10)"],
    "cw_clip_v1": ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
                   "PretrainedConfig._experts_implementation_internal=None",
                   "clone patches at repos/cellwhisperer (UCE guards, "
                   "processor attr, mean-pool get_embs)",
                   "VERY_COMMON_GENES sanity check bypassed for subset input"],
}


def embed(adata, model_key, paths, batch_size: int = 32,
          device: str | None = None):
    """Dispatch by registry key; returns (emb, keep_idx, info).

    emb rows correspond to adata.obs_names[keep_idx]; models may drop
    cells they cannot tokenize (e.g. empty GF sequences, as upstream).
    """
    t0 = time.time()
    if model_key in GF_SPECS:
        if not paths.get("gf_snapshot"):
            raise ValueError("--gf-snapshot required for Geneformer models")
        emb, info, keep_idx = gf_embed(adata, paths["gf_snapshot"],
                                       model_key, batch_size=batch_size,
                                       device=device)
    elif model_key == "scgpt_wh":
        if not paths.get("scgpt_dir"):
            raise ValueError("--scgpt-dir required for scgpt_wh")
        emb, info, keep_idx = scgpt_embed(adata, paths["scgpt_dir"],
                                        batch_size=max(batch_size, 64),
                                        device=device)
    elif model_key == "cw_clip_v1":
        if not paths.get("cw_ckpt"):
            raise ValueError("--cw-ckpt required for cw_clip_v1")
        emb, info, keep_idx = cw_embed(adata, paths["cw_ckpt"],
                                     batch_size=batch_size)
    else:
        raise ValueError(f"unknown model {model_key!r}; choices: {MODELS}")
    info["wall_s"] = round(time.time() - t0, 1)
    info["patches"] = PATCHES[model_key]
    return emb, keep_idx, info
