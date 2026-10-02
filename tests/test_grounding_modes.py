"""Regression tests for M2/M4 machinery and the gene-symbol bug.

The bug: census var_names were Ensembl/positional IDs, so captions carried
tokens like "1694" instead of "CD74" — semantically empty grounding. These
tests pin the invariants that broke.
"""

import numpy as np
import pandas as pd
import anndata

from biocellai.captions import build_cell_captions, marker_table
from biocellai.data import donor_split, use_gene_symbols
from biocellai.experiment import load_dataset
from biocellai.train import TrainConfig, train_contrastive
from conftest import make_adata


def test_use_gene_symbols_maps_feature_name():
    adata = make_adata()
    symbols = adata.var_names.to_numpy().copy()
    adata.var["feature_name"] = symbols
    adata.var_names = [str(i) for i in range(adata.n_vars)]  # positional IDs
    fixed = use_gene_symbols(adata)
    assert list(fixed.var_names) == list(symbols)


def test_use_gene_symbols_noop_without_feature_name():
    adata = make_adata()
    assert use_gene_symbols(adata).var_names[0] == "GENE0"


def test_use_gene_symbols_dedupes():
    adata = make_adata()
    adata.var["feature_name"] = ["DUP"] * adata.n_vars
    fixed = use_gene_symbols(adata)
    assert len(set(fixed.var_names)) == adata.n_vars  # DUP, DUP-1, ...


def test_type_markers_caption_uses_train_markers():
    adata = make_adata()
    markers = marker_table(adata, n_markers=3)
    caps = build_cell_captions(
        adata, include_label=False, mode="type_markers", markers=markers,
    )
    assert len(caps) == adata.n_obs
    # every T cell caption carries T's marker genes, none carry the literal label
    t_cap = caps[0]
    assert all(g in t_cap for g in markers["T"])
    assert "A T cell" not in t_cap


def test_type_llm_caption_uses_descriptions():
    adata = make_adata()
    descs = {"T": "Alpha-beta lineage adaptive responder.",
             "B": "Antibody-producing lymphocyte.",
             "Mono": "Phagocytic innate sentinel."}
    caps = build_cell_captions(
        adata, include_label=False, mode="type_llm", type_descriptions=descs,
    )
    assert len(caps) == adata.n_obs
    assert "adaptive responder" in caps[0]   # first 20 cells are T
    assert "Phagocytic" in caps[-1]          # last 20 are Mono


def test_type_llm_requires_descriptions():
    import pytest

    adata = make_adata()
    with pytest.raises(ValueError, match="type_descriptions"):
        build_cell_captions(adata, include_label=False, mode="type_llm")


def test_donor_split_no_overlap():
    adata = make_adata()
    tr, te = donor_split(adata, seed=0)
    tr_d = set(adata.obs.donor_id[tr])
    te_d = set(adata.obs.donor_id[te])
    assert tr_d.isdisjoint(te_d) and tr_d | te_d == {"d0", "d1"}


def test_contrastive_returns_loss_history():
    import torch

    adata = make_adata()
    rng = np.random.default_rng(0)
    emb = torch.from_numpy(rng.normal(size=(adata.n_obs, 16)).astype(np.float32))
    cfg = TrainConfig(seed=0, epochs=5, device="cpu")
    enc, proj, hist = train_contrastive(adata.X, emb, cfg)
    assert isinstance(hist, list) and len(hist) == 5
    assert all(np.isfinite(h) for h in hist)


def test_h5ad_roundtrip_applies_gene_symbols(tmp_path):
    """The cache-hit bug: h5ad loaded with positional var_names must still
    get feature_name symbols (load_dataset applies use_gene_symbols)."""
    rng = np.random.default_rng(0)
    # preprocess requires >=200 detected genes/cell -> denser synthetic matrix
    X = rng.poisson(3.0, (60, 300)).astype(np.float32)
    var = pd.DataFrame(index=[str(i) for i in range(300)])
    var["feature_name"] = [f"SYM{i}" for i in range(300)]
    obs = pd.DataFrame({"cell_type": np.repeat(["T", "B"], 30),
                        "donor_id": np.tile(["d0", "d1"], 30)})
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    f = tmp_path / "cached.h5ad"
    adata.write_h5ad(f)
    loaded, _ = load_dataset(str(f), seed=0, max_cells=60, n_hvg=20)
    assert str(loaded.var_names[0]).startswith("SYM")
