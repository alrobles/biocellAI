import numpy as np
import torch

from biocellai.data import donor_split
from biocellai.model import CellEncoder, info_nce
from biocellai.train import (
    TrainConfig,
    linear_probe,
    metrics,
    train_contrastive,
    train_supervised,
)
from conftest import make_adata


def test_encoder_shape():
    enc = CellEncoder(100, (64, 32), 16)
    out = enc(torch.randn(8, 100))
    assert out.shape == (8, 16)


def test_info_nce_perfect_alignment_low():
    t = torch.randn(16, 32)
    loss_same = info_nce(t, t).item()
    loss_shuffled = info_nce(t, t[torch.randperm(16)]).item()
    assert loss_same < loss_shuffled


def test_supervised_learns_synthetic():
    adata = make_adata()
    X, y = adata.X, adata.obs["cell_type"].map({"T": 0, "B": 1, "Mono": 2}).to_numpy()
    cfg = TrainConfig(epochs=30, seed=0)
    enc, head, hist = train_supervised(X, y, cfg)
    assert len(hist) == 30 and hist[-1] < hist[0]
    import torch.nn.functional as F

    with torch.no_grad():
        pred = head(enc(torch.from_numpy(X))).argmax(-1).numpy()
    acc = (pred == y).mean()
    assert acc > 0.8  # blocky synthetic data is easy


def test_contrastive_trains_and_probes():
    adata = make_adata()
    X = torch.from_numpy(adata.X)
    y = adata.obs["cell_type"].map({"T": 0, "B": 1, "Mono": 2}).to_numpy()
    # fake text embeddings: one distinct direction per class, repeated per cell
    rng = np.random.default_rng(1)
    centroids = torch.from_numpy(rng.normal(size=(3, 32)).astype(np.float32))
    T = centroids[y] + 0.01 * torch.randn(len(y), 32)
    cfg = TrainConfig(epochs=30, seed=0)
    enc, proj, hist = train_contrastive(adata.X, T, cfg)
    assert len(hist) == 30
    pred = linear_probe(enc, adata.X, y, adata.X, cfg)
    acc = (pred == y).mean()
    assert acc > 0.7


def test_metrics_keys():
    m = metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert set(m) == {"macro_f1", "balanced_acc"}


def test_donor_split_no_overlap():
    adata = make_adata()
    tr, te = donor_split(adata, seed=0)
    donors_tr = set(adata.obs["donor_id"][tr])
    donors_te = set(adata.obs["donor_id"][te])
    assert donors_tr.isdisjoint(donors_te)
    assert tr.sum() + te.sum() == adata.n_obs
