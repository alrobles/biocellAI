"""V2 experiment runner: equivalent-supervision arm matrix (ADR-011 R3).

All arms see the same cells, the same donor split, the same encoder capacity,
and the same update budget. Only the *training signal* differs:

  B0  composition (+ optional covariates) -> ridge          (no encoder)
  B1  pseudobulk -> train-fit PCA -> ridge                  (no encoder)
  B2  supervised MLP: cell-type CE + donor-pathology MSE head
  T0  contrastive vs one-hot class prototypes               (identity, no semantics)
  T1  contrastive vs fixed random prototypes                (geometry, no biology)
  T2/T3/T4 contrastive vs text captions (manual / retrieval / RAG+synthesis)
  N1  B2 with donor->pathology labels permuted within train (supervision null)
  N2  text arm with a permuted class->caption map           (semantics null)

Evaluation: identity via linear probe + zero-shot/head F1 on held-out cells;
donor pathology via train-fit ridge on donor-pooled embeddings with
per-donor predictions exported for paired inference (paired_rho_difference).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import shuffle_training_donor_labels
from .model import CellEncoder
from .progression import (
    donor_embeddings,
    pseudobulk_donor,
    ridge_predictions,
)
from .train import (
    TrainConfig,
    _to_tensor,
    class_prototypes,
    encode_labels,
    linear_probe,
    metrics,
    train_contrastive,
    train_supervised,
    zeroshot_predict,
)

CORE_ARMS = ["B0", "B1", "B2", "T0", "T1", "N1"]
TEXT_ARMS = ["T2", "T3", "T4", "N2"]
ALL_ARMS = CORE_ARMS + TEXT_ARMS


@dataclass
class V2RunConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    label_col: str = "cell_type"
    pathology_col: str | None = None
    covariates: tuple[str, ...] = ()
    proto_dim: int = 384
    ridge_alpha: float = 1.0
    seed: int = 0


def donor_target(obs: pd.DataFrame, col: str) -> pd.Series:
    """One pathology value per donor; reject non-donor-constant columns."""
    donor_ids = obs["donor_id"].astype(str)
    grouped = obs.assign(_d=donor_ids).groupby("_d")[col]
    if not (grouped.nunique() <= 1).all():
        raise ValueError(f"{col} is not donor-constant")
    return grouped.first().astype(float)


def composition_features(adata, label_col: str, train_mask: np.ndarray,
                         covariates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per-donor cell-type fractions; vocabulary learned from train cells only."""
    vocab = sorted(adata.obs[label_col].astype(str)[train_mask].unique())
    onehot = pd.get_dummies(adata.obs[label_col].astype(str)).reindex(
        columns=vocab, fill_value=0)
    comp = donor_embeddings(onehot.to_numpy(dtype=np.float32),
                            adata.obs["donor_id"].astype(str))
    for cov in covariates:
        comp[cov] = donor_target(adata.obs, cov)
    return comp


def train_supervised_dual(X_tr, y_type: np.ndarray, y_path: np.ndarray,
                          cfg: TrainConfig):
    """B2/N1: encoder + type CE head + pathology regression head — the
    non-text equivalent of the dual-head text objective."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X = _to_tensor(X_tr).to(cfg.device)
    yt = torch.from_numpy(y_type).long().to(cfg.device)
    yp = torch.from_numpy(np.asarray(y_path, dtype=np.float32)).to(cfg.device)
    enc = CellEncoder(X.shape[1], cfg.hidden, cfg.embed_dim, cfg.dropout).to(cfg.device)
    head_t = nn.Linear(cfg.embed_dim, int(yt.max().item()) + 1).to(cfg.device)
    head_p = nn.Linear(cfg.embed_dim, 1).to(cfg.device)
    params = (list(enc.parameters()) + list(head_t.parameters())
              + list(head_p.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    history = []
    enc.train()
    for _ in range(cfg.epochs):
        ep_loss, nb = 0.0, 0
        for b in _batches(X.shape[0], cfg.batch_size, rng):
            opt.zero_grad()
            h = enc(X[b])
            loss = F.cross_entropy(head_t(h), yt[b]) + F.mse_loss(
                head_p(h).squeeze(-1), yp[b])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        history.append(ep_loss / nb)
    enc.eval()
    return enc, head_t, head_p, history


def _batches(n: int, bs: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def _encode_all(enc, X, cfg) -> np.ndarray:
    with torch.no_grad():
        return enc(_to_tensor(X).to(cfg.device)).cpu().numpy()


def _donor_ridge(feats: pd.DataFrame, target: pd.Series, train_donors,
                 test_donors, alpha: float):
    """Train-fit ridge on donor-level features -> metrics + predictions."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    pred = ridge_predictions(feats, target, train_donors, test_donors, alpha)
    rho = spearmanr(pred.predicted, pred.observed).statistic
    m = {"path_rho": float(rho),
         "path_r2": float(r2_score(pred.observed, pred.predicted)),
         "path_mae": float(mean_absolute_error(pred.observed, pred.predicted))}
    return m, pred.reset_index().rename(columns={"donor_id": "donor_id",
                                               "index": "donor_id"})


def run_fold(adata, arms: list[str], cfg: V2RunConfig,
             text_targets: dict[str, dict] | None = None):
    """Run one prepared fold (obs['is_test'] from prepare_donor_fold).

    text_targets maps a text arm (T2/T3/T4/N2) to {"bank": tensor} where
    rows of `bank` are aligned to the sorted class categories of label_col.
    N2's bank must already contain the permuted class->caption mapping.
    """
    text_targets = text_targets or {}
    unknown = set(arms) - set(ALL_ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    missing = [a for a in arms if a in TEXT_ARMS and a not in text_targets]
    if missing:
        raise ValueError(f"arms {missing} require text_targets (--captions)")

    is_test = adata.obs["is_test"].to_numpy()
    train_mask, test_mask = ~is_test, is_test
    train_donors = sorted(adata.obs["donor_id"].astype(str)[train_mask].unique())
    test_donors = sorted(adata.obs["donor_id"].astype(str)[test_mask].unique())
    if not train_donors or not test_donors:
        raise ValueError("fold has no train or test donors")

    y, cats = encode_labels(adata, cfg.label_col)
    X = adata.X
    target = (donor_target(adata.obs, cfg.pathology_col)
              if cfg.pathology_col is not None else None)
    if "N1" in arms and target is None:
        raise ValueError(
            "N1 permutes pathology labels; --pathology-col is required")
    tc = cfg.train

    rows, predictions, states = [], {}, {}

    def record(arm, m, pred=None, state=None):
        m.update({"arm": arm, "n_train_donors": len(train_donors),
                  "n_test_donors": len(test_donors)})
        rows.append(m)
        if pred is not None:
            predictions[arm] = pred
        if state is not None:
            states[arm] = state

    def eval_encoder(arm, enc, proj=None, bank=None, head=None):
        emb = _encode_all(enc, X, tc)
        m = {"identity_f1_probe": np.nan, "identity_f1_zeroshot": np.nan,
             "identity_eval": "", "path_rho": np.nan, "path_r2": np.nan,
             "path_mae": np.nan}
        if head is not None:  # supervised: head predictions on test cells
            with torch.no_grad():
                preds = head(enc(_to_tensor(X[test_mask]).to(tc.device))
                             ).argmax(-1).cpu().numpy()
            m["identity_f1_probe"] = metrics(y[test_mask], preds)["macro_f1"]
            m["identity_eval"] = "head"
        else:
            probe_preds = linear_probe(enc, X[train_mask], y[train_mask],
                                       X[test_mask], tc)
            m["identity_f1_probe"] = metrics(y[test_mask],
                                             probe_preds)["macro_f1"]
            m["identity_eval"] = "probe"
            if proj is not None and bank is not None:
                zs = zeroshot_predict(enc, proj, X[test_mask], bank, tc)
                m["identity_f1_zeroshot"] = metrics(y[test_mask], zs)["macro_f1"]
        pred = None
        if target is not None:
            demb = donor_embeddings(emb, adata.obs["donor_id"].astype(str))
            pm, pred = _donor_ridge(demb, target, train_donors, test_donors,
                                    cfg.ridge_alpha)
            m.update(pm)
        state = {"enc": enc.state_dict()}
        if proj is not None:
            state["proj"] = proj.state_dict()
        record(arm, m, pred, state)

    for arm in arms:
        if arm == "B0":
            feats = composition_features(adata, cfg.label_col, train_mask,
                                         cfg.covariates)
            m = {"identity_f1_probe": np.nan, "identity_f1_zeroshot": np.nan,
                 "identity_eval": "", "path_rho": np.nan, "path_r2": np.nan,
                 "path_mae": np.nan}
            pred = None
            if target is not None:
                pm, pred = _donor_ridge(feats, target, train_donors,
                                        test_donors, cfg.ridge_alpha)
                m.update(pm)
            record(arm, m, pred)
            continue

        if arm == "B1":
            pb = pseudobulk_donor(X, adata.obs["donor_id"].astype(str),
                                  seed=cfg.seed, train_donors=train_donors)
            m = {"identity_f1_probe": np.nan, "identity_f1_zeroshot": np.nan,
                 "identity_eval": "", "path_rho": np.nan, "path_r2": np.nan,
                 "path_mae": np.nan}
            pred = None
            if target is not None:
                pm, pred = _donor_ridge(pb, target, train_donors, test_donors,
                                        cfg.ridge_alpha)
                m.update(pm)
            record(arm, m, pred)
            continue

        if arm in ("B2", "N1"):
            if target is not None:
                obs2 = adata.obs.copy()
                obs2["_path"] = adata.obs["donor_id"].astype(str).map(target)
                if arm == "N1":
                    obs2["_path"] = shuffle_training_donor_labels(
                        obs2, "_path", train_donors, seed=cfg.seed + 777)
                y_path = obs2["_path"].to_numpy(dtype=np.float32)
                enc, head_t, head_p, _ = train_supervised_dual(
                    X[train_mask], y[train_mask], y_path[train_mask], tc)
            else:
                enc, head_t, _ = train_supervised(
                    X[train_mask], y[train_mask], tc)
            eval_encoder(arm, enc, head=head_t)
            continue

        if arm in ("T0", "T1"):
            labels = adata.obs[cfg.label_col].astype(str).to_numpy()
            if arm == "T0":
                per_cell, bank, _ = class_prototypes(
                    labels, "onehot", max(len(cats), cfg.proto_dim),
                    cfg.seed + 11)
            else:
                per_cell, bank, _ = class_prototypes(
                    labels, "random", cfg.proto_dim, cfg.seed + 13)
            enc, proj, _ = train_contrastive(X[train_mask],
                                             per_cell[train_mask], tc)
            eval_encoder(arm, enc, proj=proj, bank=bank)
            continue

        # T2/T3/T4/N2 — text-embedding targets supplied by the caller
        bank = text_targets[arm]["bank"]
        positions = {c: i for i, c in enumerate(cats)}
        per_cell = bank[[positions[c]
                         for c in adata.obs[cfg.label_col].astype(str)]]
        enc, proj, _ = train_contrastive(X[train_mask],
                                         per_cell[train_mask], tc)
        eval_encoder(arm, enc, proj=proj, bank=bank)

    return pd.DataFrame(rows), predictions, states


def write_run(out: Path, metrics_df, predictions, states, manifest: dict):
    """Write run artifacts into an already-reserved output directory
    (see revalidation.reserve_output — refuses existing paths)."""
    if not out.is_dir():
        raise ValueError(f"{out} is not a reserved output directory")
    metrics_df.to_csv(out / "metrics.csv", index=False)
    for arm, pred in predictions.items():
        pred.to_csv(out / f"predictions_{arm}.csv", index=False)
    for arm, state in states.items():
        torch.save(state, out / f"ckpt_{arm}.pt")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
