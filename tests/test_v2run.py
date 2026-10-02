import json

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from biocellai import v2run
from biocellai.data import prepare_donor_fold, split_donor_ids
from biocellai.revalidation import reserve_output
from biocellai.train import TrainConfig


def synthetic_fold(seed=0):
    """12 donors, 3 cell types, donor-constant pathology score."""
    rng = np.random.default_rng(seed)
    n_donors, cells_per, n_genes = 12, 15, 200
    n = n_donors * cells_per
    donor = np.repeat([f"d{i}" for i in range(n_donors)], cells_per)
    # per-donor random type mix -> pathology score driven by T-cell fraction,
    # giving B0 a real composition signal to recover
    ctype = np.empty(n, dtype=object)
    tfrac = np.zeros(n_donors)
    for i in range(n_donors):
        probs = rng.dirichlet([2, 2, 2])
        types = rng.choice(["Tcell", "Bcell", "Mono"], size=cells_per, p=probs)
        ctype[i * cells_per:(i + 1) * cells_per] = types
        tfrac[i] = (types == "Tcell").mean()
    path = np.repeat(3 * tfrac + rng.normal(scale=0.05, size=n_donors),
                     cells_per)
    X = rng.poisson(3.0, (n, n_genes)).astype(np.float32)
    # give cell types a weak signal so identity metrics are meaningful
    for i, t in enumerate(["Tcell", "Bcell", "Mono"]):
        X[ctype == t, i * 50:(i + 1) * 50] += 5
    a = AnnData(X=X, obs=pd.DataFrame(
        {"donor_id": donor, "cell_type": ctype, "path_score": path},
        index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    train, test = split_donor_ids(a.obs["donor_id"], seed=seed)
    return (prepare_donor_fold(a, train, test, n_hvg=100, min_genes=1,
                               min_cells=3), list(test))


def cfg(seed=0):
    return v2run.V2RunConfig(
        train=TrainConfig(epochs=3, batch_size=64, seed=seed,
                          hidden=(32,), embed_dim=16, probe_epochs=20),
        pathology_col="path_score", proto_dim=16, seed=seed)


def test_core_arms_run_and_write(tmp_path):
    fold, test_donors = synthetic_fold()
    arms = ["B0", "B1", "B2", "T0", "T1", "N1"]
    metrics_df, preds, states = v2run.run_fold(fold, arms, cfg())
    assert list(metrics_df["arm"]) == arms
    enc_arms = ["B2", "T0", "T1", "N1"]
    for arm in enc_arms:
        row = metrics_df.set_index("arm").loc[arm]
        assert np.isfinite(row["identity_f1_probe"])
        assert np.isfinite(row["path_rho"])
    for arm in ["B0", "B1"]:
        assert np.isfinite(metrics_df.set_index("arm").loc[arm, "path_rho"])
    for arm in ["T0", "T1"]:
        assert np.isfinite(
            metrics_df.set_index("arm").loc[arm, "identity_f1_zeroshot"])
    for arm, pred in preds.items():
        assert len(pred) == len(test_donors)
        assert set(pred["donor_id"]) == set(test_donors)
    out = reserve_output(tmp_path / "run")
    v2run.write_run(out, metrics_df, preds, states, {"test": True})
    assert (out / "metrics.csv").exists()
    assert (out / "manifest.json").exists()
    for arm in enc_arms:
        assert (out / f"predictions_{arm}.csv").exists()
        assert (out / f"ckpt_{arm}.pt").exists()
    with pytest.raises((FileExistsError, ValueError)):
        reserve_output(out)


def test_text_arms_use_supplied_banks(tmp_path):
    fold, _ = synthetic_fold()
    cats = sorted(fold.obs["cell_type"].unique())
    rng = np.random.default_rng(5)
    bank = torch.from_numpy(
        rng.normal(size=(len(cats), 16)).astype(np.float32))
    targets = {"T2": {"bank": bank},
               "N2": {"bank": bank[torch.tensor([2, 0, 1])]}}
    metrics_df, preds, states = v2run.run_fold(
        fold, ["T2", "N2"], cfg(), text_targets=targets)
    row = metrics_df.set_index("arm")
    for arm in ["T2", "N2"]:
        assert np.isfinite(row.loc[arm, "identity_f1_probe"])
        assert np.isfinite(row.loc[arm, "identity_f1_zeroshot"])


def test_text_arm_without_captions_rejected():
    fold, _ = synthetic_fold()
    with pytest.raises(ValueError, match="text_targets"):
        v2run.run_fold(fold, ["T3"], cfg())


def test_unknown_arm_rejected():
    fold, _ = synthetic_fold()
    with pytest.raises(ValueError, match="unknown"):
        v2run.run_fold(fold, ["ZZ"], cfg())


def test_deterministic_same_seed():
    fold, _ = synthetic_fold()
    m1, _, _ = v2run.run_fold(fold, ["B2", "T0"], cfg(seed=0))
    m2, _, _ = v2run.run_fold(fold, ["B2", "T0"], cfg(seed=0))
    pd.testing.assert_frame_equal(m1, m2)


def test_no_pathology_col_skips_path_metrics():
    fold, _ = synthetic_fold()
    c = v2run.V2RunConfig(train=TrainConfig(epochs=2, batch_size=64,
                                            hidden=(32,), embed_dim=16,
                                            probe_epochs=10),
                          proto_dim=16)
    metrics_df, preds, _ = v2run.run_fold(fold, ["B0", "B2"], c)
    assert metrics_df["path_rho"].isna().all()
    assert preds == {}
    with pytest.raises(ValueError, match="pathology"):
        v2run.run_fold(fold, ["N1"], c)
