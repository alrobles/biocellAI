"""R2-TRANSFER tests — strict/refit separation on synthetic cohorts."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix

from biocellai.model import CellEncoder
from biocellai.transfer import (
    target_matrix_on_source_genes,
    transfer_arm,
)


def _source_fold(n_genes=20):
    """Synthetic source fold: 8 donors x 6 cells, pathology = donor index."""
    rng = np.random.default_rng(0)
    donors = np.repeat([f"s{i}" for i in range(8)], 6)
    is_test = np.isin(donors, [f"s{i}" for i in range(6, 8)])
    obs = pd.DataFrame({
        "donor_id": donors,
        "is_test": is_test,
        "cell_type": np.tile(["A", "B"], 24),
        "path": np.repeat(np.arange(8, dtype=float), 6),
    })
    return AnnData(X=rng.normal(size=(48, n_genes)).astype(np.float32),
                   obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))


def _target(n_genes=25, path_values=None):
    """Target raw counts + obs; shares genes g0..g19 with the source fold."""
    rng = np.random.default_rng(1)
    donors = np.repeat([f"t{i}" for i in range(10)], 5)
    path = (np.repeat(np.arange(10, dtype=float), 5)
            if path_values is None else np.asarray(path_values))
    obs = pd.DataFrame({
        "donor_id": donors,
        "is_test": np.isin(donors, ["t7", "t8", "t9"]),
        "cell_type": np.tile(["A", "C"], 25),
        "tpath": path,
    })
    raw = AnnData(
        X=csr_matrix(rng.poisson(2.0, (50, n_genes)).astype(np.float32)),
        obs=obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    return raw, obs


def _run_dir(tmp_path, n_genes=20):
    run = tmp_path / "run"
    run.mkdir()
    enc = CellEncoder(n_genes, (8, 4), 6)
    torch.save({"enc": enc.state_dict()}, run / "ckpt_B2.pt")
    (run / "manifest.json").write_text(json.dumps({
        "config": {"train": {"hidden": "(8, 4)", "embed_dim": "6"},
                   "pathology_col": "path", "ridge_alpha": 1.0}}))
    return run


def test_target_matrix_normalizes_and_zero_fills():
    raw, obs = _target()
    genes = list(raw.var_names[:20]) + ["NOT_IN_RAW"]
    X, stats = target_matrix_on_source_genes(raw, obs.index, genes)
    assert X.shape == (50, 21)
    assert stats["n_genes_matched"] == 20
    assert stats["n_genes_zero_filled"] == 1
    dense = raw.X.toarray()
    expected = np.log1p(dense / dense.sum(1, keepdims=True) * 1e4)[:, :20]
    np.testing.assert_allclose(X[:, :20].toarray(), expected, rtol=1e-6)
    assert (X[:, 20].toarray() == 0).all()


def test_target_matrix_rejects_missing_cells():
    raw, _ = _target()
    with pytest.raises(ValueError, match="missing"):
        target_matrix_on_source_genes(raw, ["nope"], list(raw.var_names))


def test_strict_predictions_do_not_use_target_labels(tmp_path):
    src = _source_fold()
    run = _run_dir(tmp_path, src.n_vars)
    raw, obs = _target()
    tgt_X, _ = target_matrix_on_source_genes(
        raw, obs.index, list(src.var_names.astype(str)))

    rows1, preds1, _ = transfer_arm(
        "B1", src, run, "path", "cell_type", 1.0, 0, "cpu",
        tgt_X, obs, "tpath")
    perm = obs.copy()
    dvals = np.arange(10, dtype=float)
    permuted = np.random.default_rng(7).permutation(dvals)
    perm["tpath"] = perm["donor_id"].map(
        dict(zip([f"t{i}" for i in range(10)], permuted)))
    rows2, preds2, _ = transfer_arm(
        "B1", src, run, "path", "cell_type", 1.0, 0, "cpu",
        tgt_X, perm, "tpath")

    np.testing.assert_array_equal(
        preds1["strict"]["predicted"], preds2["strict"]["predicted"])
    r1 = {r["eval_subset"]: r["rho"]
          for r in rows1 if r["mode"] == "strict"}
    assert set(r1) == {"all_donors", "test_donors"}
    # refit uses target labels -> observed differs under permutation
    assert not preds1["refit"]["observed"].equals(preds2["refit"]["observed"])


def test_encoder_arm_transfers_and_verifies(tmp_path):
    src = _source_fold()
    run = _run_dir(tmp_path, src.n_vars)
    raw, obs = _target()
    tgt_X, _ = target_matrix_on_source_genes(
        raw, obs.index, list(src.var_names.astype(str)))
    rows, preds, info = transfer_arm(
        "B2", src, run, "path", "cell_type", 1.0, 0, "cpu",
        tgt_X, obs, "tpath")
    modes = {(r["arm"], r["mode"]) for r in rows}
    assert ("B2", "strict") in modes and ("B2", "refit") in modes
    assert info["n_target_donors"] == 10
    strict = preds["strict"]
    assert len(strict) == 10 and strict["predicted"].notna().all()


def test_b0_vocab_coverage_reported(tmp_path):
    src = _source_fold()
    run = _run_dir(tmp_path, src.n_vars)
    raw, obs = _target()
    tgt_X, _ = target_matrix_on_source_genes(
        raw, obs.index, list(src.var_names.astype(str)))
    rows, preds, info = transfer_arm(
        "B0", src, run, "path", "cell_type", 1.0, 0, "cpu",
        tgt_X, obs, "tpath")
    # target cells carry types {A, C}; source vocab {A, B} -> 50% coverage
    assert info["label_vocab_coverage"] == pytest.approx(0.5)
    assert info["n_vocab"] == 2
