"""R2-TRANSFER — strict cross-cohort transfer vs calibrated orientation.

ADR-011: "Strict transfer: fix the encoder, normalization, axis/predictor, and
sign in the source cohort. Orientation using target labels is a separate
calibrated analysis." Gate: rho >= 0.35 on the target pathology axis.

STRICT mode: every component is fitted on the source cohort only — the donor
encoder (ckpt_<arm>.pt from the source run), the feature construction (source
gene list, source cell-type vocabulary, train-only PCA), and the donor-level
readout (StandardScaler + Ridge refit on source TRAIN donors). The target
cohort contributes cells grouped by donor_id; its labels are used only for
evaluation. The evaluation universe is the frozen target-fold cell set
(obs_names); expression values are recomputed from the cohort's verified
counts extract with the identical per-cell transform (total-1e4 then log1p),
projected onto the SOURCE gene list — genes absent from the target extract
are zero-filled and counted in the manifest.

REFIT mode (calibrated/orientation): identical source features, but the donor
readout is fit on TARGET train donors and evaluated on TARGET test donors.
This is supervised target-cohort orientation — reported separately, never
mixed into the strict table.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .model import CellEncoder
from .progression import donor_embeddings
from .v2run import composition_features, donor_target

ENCODER_ARMS = ("B2", "N1", "T0", "T1", "T2", "T3", "T4", "N2")
FEATURE_ARMS = ("B0", "B1")
ALL_TRANSFER_ARMS = FEATURE_ARMS + ENCODER_ARMS


def _csr_rows(X, rows):
    """Vectorized CSR row-gather (same approach as v2_extract._csr_rows)."""
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    rows = np.asarray(rows, dtype=np.int64)
    indptr = X.indptr.astype(np.int64, copy=False)
    nnz = indptr[rows + 1] - indptr[rows]
    new_indptr = np.empty(len(rows) + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(nnz, out=new_indptr[1:])
    gather = np.repeat(indptr[rows] - new_indptr[:-1], nnz) + np.arange(
        new_indptr[-1], dtype=np.int64)
    return sp.csr_matrix(
        (X.data[gather], X.indices[gather], new_indptr),
        shape=(len(rows), X.shape[1]), dtype=X.dtype)


def target_matrix_on_source_genes(raw, fold_obs_names, source_genes):
    """Recompute the source transform on target counts, in source gene order.

    raw: AnnData with X = verified counts (full transcriptome) and obs_names
         covering fold_obs_names.
    Returns (csr_matrix n_cells x len(source_genes), stats dict).
    """
    fold_obs_names = pd.Index(
        np.asarray(list(fold_obs_names), dtype=str))
    raw_names = pd.Index(np.asarray(raw.obs_names.astype(str)))
    pos = raw_names.get_indexer(fold_obs_names)
    if (pos < 0).any():
        raise ValueError(
            f"{int((pos < 0).sum())} fold cells missing from raw extract")
    X = _csr_rows(raw.X, pos)
    totals = np.asarray(X.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError("target cells with zero total counts")
    X = (sp.diags(1e4 / totals) @ X).tocsr()
    X.data = np.log1p(X.data)

    raw_genes = pd.Index(raw.var_names.astype(str))
    gene_pos = raw_genes.get_indexer(pd.Index(map(str, source_genes)))
    matched = gene_pos >= 0
    proj = sp.csr_matrix(
        (np.ones(int(matched.sum()), dtype=np.float32),
         (gene_pos[matched], np.where(matched)[0])),
        shape=(X.shape[1], len(source_genes)))
    out = (X @ proj).tocsr()
    stats = {
        "n_source_genes": int(len(source_genes)),
        "n_genes_matched": int(matched.sum()),
        "n_genes_zero_filled": int((~matched).sum()),
        "gene_coverage": float(matched.mean()),
        "n_target_cells": int(X.shape[0]),
        "transform": "full_transcriptome_total_10000_then_log1p",
    }
    return out.tocsr(), stats


def load_encoder(ckpt_path: Path, n_input: int, hidden, embed_dim: int,
                 dropout: float = 0.0, device: str = "cpu") -> CellEncoder:
    """Rebuild a CellEncoder from a run checkpoint (state dict under 'enc')."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    enc = CellEncoder(n_input, tuple(hidden), embed_dim, dropout)
    enc.load_state_dict(state.get("enc", state))
    enc.eval().to(device)
    return enc


def encode_cells(enc, X, device: str = "cpu", batch: int = 8192) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = X[i:i + batch]
            if sp.issparse(xb):
                xb = xb.toarray()
            outs.append(
                enc(torch.from_numpy(np.asarray(xb, dtype=np.float32))
                    .to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _ridge_fit(feats, target, donors, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(feats.loc[list(donors)].to_numpy(),
          target.loc[list(donors)].to_numpy())
    return m


def _metrics(pred: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score
    if len(pred) < 3:
        return {"rho": np.nan, "r2": np.nan, "mae": np.nan, "n": len(pred)}
    return {"rho": float(spearmanr(pred.predicted, pred.observed).statistic),
            "r2": float(r2_score(pred.observed, pred.predicted)),
            "mae": float(mean_absolute_error(pred.observed, pred.predicted)),
            "n": int(len(pred))}


class _SourceContext:
    """Source-fitted feature extractor + readout for one arm."""

    def __init__(self, arm, adata, run_dir, label_col, path_col,
                 ridge_alpha, seed, device):
        self.arm = arm
        is_test = adata.obs["is_test"].to_numpy()
        self.train_mask = ~is_test
        donors = adata.obs["donor_id"].astype(str)
        self.train_donors = sorted(donors[self.train_mask].unique())
        self.test_donors = sorted(donors[is_test].unique())
        self.target = donor_target(adata.obs, path_col)
        self.donors = donors
        self.enc = None
        self.pca = None
        self.vocab = None
        self.label_col = label_col
        if arm == "B0":
            self.vocab = sorted(
                adata.obs[label_col].astype(str)[self.train_mask].unique())
        elif arm == "B1":
            self.seed = seed
        else:
            enc_cfg = json.loads(
                (Path(run_dir) / "manifest.json").read_text()
            )["config"]["train"]
            self.enc = load_encoder(
                Path(run_dir) / f"ckpt_{arm}.pt", adata.n_vars,
                ast.literal_eval(enc_cfg["hidden"]),
                int(enc_cfg["embed_dim"]), device=device)
        self.device = device
        self.alpha = ridge_alpha
        self.src_feats = self.feats_from(adata.X, donors,
                                         adata.obs[label_col])
        self.readout = _ridge_fit(
            self.src_feats, self.target, self.train_donors, ridge_alpha)

    def feats_from(self, X, donors, labels=None):
        """Donor-level features for cells whose X is already in the SOURCE
        gene space (source fold X, or target counts transformed+projected)."""
        if self.arm == "B0":
            if labels is None:
                raise ValueError("B0 requires cell-type labels")
            onehot = pd.get_dummies(labels.astype(str)).reindex(
                columns=self.vocab, fill_value=0)
            return donor_embeddings(
                onehot.to_numpy(dtype=np.float32), donors)
        if self.arm == "B1":
            pb = donor_embeddings(
                np.asarray(X.todense() if hasattr(X, "todense") else X,
                           dtype=np.float32), donors)
            if self.pca is None:
                from sklearn.decomposition import PCA
                self.pca = PCA(
                    n_components=min(20, len(self.train_donors) - 1,
                                     pb.shape[1]),
                    random_state=self.seed).fit(pb.loc[self.train_donors])
            return pd.DataFrame(self.pca.transform(pb), index=pb.index)
        emb = encode_cells(self.enc, X, device=self.device)
        return donor_embeddings(emb, donors)


def transfer_arm(arm: str, src_adata, src_run_dir: Path, src_path_col: str,
                 label_col: str, ridge_alpha: float, seed: int, device: str,
                 tgt_X, tgt_obs: pd.DataFrame, tgt_path_col: str):
    """One arm in strict and refit modes -> (rows, predictions, info)."""
    ctx = _SourceContext(arm, src_adata, src_run_dir, label_col,
                         src_path_col, ridge_alpha, seed, device)

    verification = {}
    ref_pred_path = Path(src_run_dir) / f"predictions_{arm}.csv"
    if ref_pred_path.exists():
        ref = pd.read_csv(ref_pred_path, index_col=0)
        test_pred = pd.Series(
            ctx.readout.predict(
                ctx.src_feats.loc[ctx.test_donors].to_numpy()),
            index=pd.Index(ctx.test_donors, name="donor_id"))
        common = ref.index.intersection(test_pred.index)
        if len(common) >= 3:
            verification["source_test_max_abs_diff"] = float(
                np.abs(ref.loc[common, "predicted"].to_numpy()
                       - test_pred.loc[common].to_numpy()).max())

    tgt_obs = tgt_obs.copy()
    tgt_obs["donor_id"] = tgt_obs["donor_id"].astype(str)
    tgt_feats = ctx.feats_from(tgt_X, tgt_obs["donor_id"],
                               tgt_obs.get(label_col))
    tgt_feats = tgt_feats.reindex(columns=ctx.src_feats.columns,
                                  fill_value=0.0)

    tgt_target = donor_target(tgt_obs, tgt_path_col)
    if "is_test" in tgt_obs:
        split = tgt_obs.groupby("donor_id")["is_test"].first().astype(bool)
        tgt_test_donors = sorted(split.index[split])
        tgt_train_donors = sorted(split.index[~split])
    else:
        tgt_test_donors, tgt_train_donors = [], []
    valid = tgt_target.dropna().index

    rows, preds = [], {}

    strict_pred = pd.Series(
        ctx.readout.predict(tgt_feats.loc[valid].to_numpy()),
        index=pd.Index(valid, name="donor_id"))
    df = pd.DataFrame({"observed": tgt_target.loc[valid],
                       "predicted": strict_pred})
    preds["strict"] = df.reset_index()
    for subset, donors in (("all_donors", list(valid)),
                           ("test_donors", [d for d in tgt_test_donors
                                            if d in valid])):
        rows.append(dict(arm=arm, mode="strict", eval_subset=subset,
                         **_metrics(df.loc[donors])))

    tr = [d for d in tgt_train_donors if d in valid]
    te = [d for d in tgt_test_donors if d in valid]
    if len(tr) >= 5 and len(te) >= 3:
        model = _ridge_fit(tgt_feats, tgt_target, tr, ridge_alpha)
        rdf = pd.DataFrame({"observed": tgt_target.loc[te],
                            "predicted": pd.Series(
                                model.predict(tgt_feats.loc[te].to_numpy()),
                                index=te)})
        preds["refit"] = rdf.reset_index()
        rows.append(dict(arm=arm, mode="refit", eval_subset="test_donors",
                         **_metrics(rdf)))

    info = {"verification": verification,
            "n_target_donors": int(len(valid)),
            "n_target_test_donors": int(len(tgt_test_donors))}
    if arm == "B0":
        info["label_vocab_coverage"] = float(
            tgt_obs[label_col].astype(str).isin(ctx.vocab).mean())
        info["n_vocab"] = len(ctx.vocab)
    return rows, preds, info
