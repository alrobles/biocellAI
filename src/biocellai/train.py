"""Training + evaluation for the M1 grounding benchmark.

Arms (identical encoder capacity, ADR-002):
  - supervised  : encoder + linear head, cross-entropy end to end
  - contrastive : encoder aligned to frozen caption embeddings via InfoNCE,
                  then evaluated by linear probe AND zero-shot nearest-class
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

from .model import CellEncoder, info_nce


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 256
    seed: int = 0
    hidden: tuple[int, ...] = (256, 128)
    embed_dim: int = 64
    dropout: float = 0.1
    temperature: float = 0.07
    device: str = "cpu"
    probe_epochs: int = 200


def _batches(n: int, bs: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def _to_tensor(X) -> torch.Tensor:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return torch.from_numpy(np.asarray(X, dtype=np.float32))


def class_prototypes(labels, mode: str, dim: int, seed: int):
    labels = np.asarray(labels).astype(str)
    cats = sorted(np.unique(labels))
    if not cats or dim < 1:
        raise ValueError("nonempty labels and positive prototype dimension required")
    if mode == "onehot":
        if dim < len(cats):
            raise ValueError("onehot dimension must be at least the number of classes")
        bank = np.eye(len(cats), dim, dtype=np.float32)
    elif mode == "random":
        bank = np.random.default_rng(seed).normal(size=(len(cats), dim)).astype(np.float32)
        bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    else:
        raise ValueError("prototype mode must be onehot or random")
    positions = {label: i for i, label in enumerate(cats)}
    ids = [positions[label] for label in labels]
    return torch.from_numpy(bank[ids]), torch.from_numpy(bank), cats


def train_supervised(X_tr, y_tr: np.ndarray, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X = _to_tensor(X_tr).to(cfg.device)
    y = torch.from_numpy(y_tr).long().to(cfg.device)

    enc = CellEncoder(X.shape[1], cfg.hidden, cfg.embed_dim, cfg.dropout).to(cfg.device)
    head = nn.Linear(cfg.embed_dim, int(y.max().item()) + 1).to(cfg.device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=cfg.lr)

    history = []
    enc.train()
    for _ in range(cfg.epochs):
        ep_loss, nb = 0.0, 0
        for b in _batches(X.shape[0], cfg.batch_size, rng):
            opt.zero_grad()
            loss = F.cross_entropy(head(enc(X[b])), y[b])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        history.append(ep_loss / nb)
    enc.eval()
    return enc, head, history


def train_contrastive(X_tr, text_emb_tr: torch.Tensor, cfg: TrainConfig,
                      levels: torch.Tensor | None = None,
                      bank: torch.Tensor | None = None,
                      ord_tau: float = 0.0):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X = _to_tensor(X_tr).to(cfg.device)
    T = text_emb_tr.to(cfg.device)
    soft = (levels is not None and bank is not None and ord_tau
            and ord_tau > 0)
    if soft:
        lv = levels.to(cfg.device)
        bank = bank.to(cfg.device)

    enc = CellEncoder(X.shape[1], cfg.hidden, cfg.embed_dim, cfg.dropout).to(cfg.device)
    proj = nn.Linear(cfg.embed_dim, T.shape[1]).to(cfg.device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=cfg.lr)

    history = []
    enc.train()
    for _ in range(cfg.epochs):
        ep_loss, nb = 0.0, 0
        for b in _batches(X.shape[0], cfg.batch_size, rng):
            opt.zero_grad()
            z = proj(enc(X[b]))
            loss = (soft_ordinal_nce(z, bank, lv[b], cfg.temperature, ord_tau)
                    if soft else info_nce(z, T[b], cfg.temperature))
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        history.append(ep_loss / nb)
    enc.eval()
    return enc, proj, history


def soft_ordinal_nce(z: torch.Tensor, bank: torch.Tensor,
                     levels: torch.Tensor, temperature: float,
                     ord_tau: float) -> torch.Tensor:
    """Soft-target InfoNCE over an ordinal caption bank.

    Instead of pulling a cell to its own level's caption only, the target
    distribution spreads mass over all level captions weighted by ordinal
    proximity: w_j = softmax(-|level_i - j| / ord_tau). Cells with level<0
    (e.g. 'nan') are excluded.
    """
    mask = levels >= 0
    if not mask.any():
        return z.sum() * 0.0
    z = z[mask]
    levels = levels[mask]
    sims = F.normalize(z, dim=-1) @ F.normalize(bank, dim=-1).T / temperature
    K = bank.shape[0]
    dist = (levels[:, None] - torch.arange(K, device=z.device)[None, :]) \
        .abs().float()
    w = F.softmax(-dist / ord_tau, dim=-1)
    return -(w * F.log_softmax(sims, dim=-1)).sum(-1).mean()


def train_contrastive_multihead(X_tr, emb_type: torch.Tensor, emb_path: torch.Tensor,
                                cfg: TrainConfig,
                                path_levels: torch.Tensor | None = None,
                                path_bank: torch.Tensor | None = None,
                                ord_tau: float = 0.0):
    """Shared encoder + two projection heads: cell-type caption head and
    donor-pathology caption head. Loss = InfoNCE_type + InfoNCE_path
    (M10: hierarchical objective — the naive A|B label concat failed
    by diluting each pathology class across subclasses).

    M11-B2: when path_levels/path_bank/ord_tau are given, the pathology
    head uses soft_ordinal_nce — a distance-weighted target over the
    ordinal level-caption bank instead of hard one-hot InfoNCE."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X = _to_tensor(X_tr).to(cfg.device)
    T_t = emb_type.to(cfg.device)
    T_p = emb_path.to(cfg.device)
    soft = (path_levels is not None and path_bank is not None
            and ord_tau and ord_tau > 0)
    if soft:
        lv = path_levels.to(cfg.device)
        bank = path_bank.to(cfg.device)

    enc = CellEncoder(X.shape[1], cfg.hidden, cfg.embed_dim, cfg.dropout).to(cfg.device)
    proj_t = nn.Linear(cfg.embed_dim, T_t.shape[1]).to(cfg.device)
    proj_p = nn.Linear(cfg.embed_dim, T_p.shape[1]).to(cfg.device)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(proj_t.parameters()) + list(proj_p.parameters()),
        lr=cfg.lr)

    history = []
    enc.train()
    for _ in range(cfg.epochs):
        ep_loss, nb = 0.0, 0
        for b in _batches(X.shape[0], cfg.batch_size, rng):
            opt.zero_grad()
            h = enc(X[b])
            z_p = proj_p(h)
            loss_p = (soft_ordinal_nce(z_p, bank, lv[b], cfg.temperature,
                                       ord_tau)
                      if soft else info_nce(z_p, T_p[b], cfg.temperature))
            loss = info_nce(proj_t(h), T_t[b], cfg.temperature) + loss_p
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        history.append(ep_loss / nb)
    enc.eval()
    return enc, proj_t, proj_p, history


def linear_probe(enc: CellEncoder, X_tr, y_tr, X_te, cfg: TrainConfig):
    """Fit a linear classifier on frozen encoder embeddings."""
    torch.manual_seed(cfg.seed)
    Xtr = _to_tensor(X_tr).to(cfg.device)
    Xte = _to_tensor(X_te).to(cfg.device)
    y = torch.from_numpy(y_tr).long().to(cfg.device)

    with torch.no_grad():
        E_tr = enc(Xtr)
        E_te = enc(Xte)
    probe = nn.Linear(cfg.embed_dim, int(y.max().item()) + 1).to(cfg.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr)
    for _ in range(cfg.probe_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(E_tr), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return probe(E_te).argmax(dim=-1).cpu().numpy()


def zeroshot_predict(enc, proj, X_te, class_text_emb: torch.Tensor, cfg: TrainConfig) -> np.ndarray:
    """Nearest class-caption in the shared embedding space."""
    Xte = _to_tensor(X_te).to(cfg.device)
    with torch.no_grad():
        E = F.normalize(proj(enc(Xte)), dim=-1)
        sims = E @ F.normalize(class_text_emb.to(cfg.device), dim=-1).T
    return sims.argmax(dim=-1).cpu().numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def encode_labels(adata, key: str = "cell_type"):
    cats = sorted(adata.obs[key].astype(str).unique())
    to_idx = {c: i for i, c in enumerate(cats)}
    y = adata.obs[key].astype(str).map(to_idx).to_numpy()
    return y, cats
