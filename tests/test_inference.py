"""R6-INFERENCE unit tests — clustered bootstrap, FDR, nulls, gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biocellai import inference as inf


def _pred(seed_donors, obs_map, noise_fn):
    """df indexed by donor with observed/predicted columns."""
    rows = {}
    for s, donors in seed_donors.items():
        rows[s] = pd.DataFrame(
            {"observed": [obs_map[d] for d in donors],
             "predicted": [obs_map[d] + noise_fn(d) for d in donors]},
            index=pd.Index(donors, name="donor_id"))
    return rows


@pytest.fixture
def pooled_preds():
    """Donors reappear across seeds; arm A tracks obs closely, B is flat."""
    rng = np.random.default_rng(7)
    donors = [f"d{i}" for i in range(30)]
    obs = {d: v for d, v in zip(donors, rng.normal(size=30))}
    per_seed = {0: donors[:15], 1: donors[10:25], 2: donors[20:]}
    a = _pred(per_seed, obs, lambda d: 0.05 * rng.normal())
    b = _pred(per_seed, obs, lambda d: rng.normal())
    return a, b, obs


def test_clustered_delta_positive(pooled_preds):
    a, b, _ = pooled_preds
    res = inf.clustered_delta_rho(a, b, n_boot=400, seed=0)
    assert res["delta_rho"] > 0
    assert res["ci_low"] < res["ci_high"]
    assert res["n_boot_valid"] > 350
    assert res["n_donors"] == 30
    assert set(res["per_seed"]) == {0, 1, 2}


def test_clustered_delta_zero_when_equal(pooled_preds):
    a, _, _ = pooled_preds
    res = inf.clustered_delta_rho(a, a, n_boot=100, seed=0)
    assert res["delta_rho"] == pytest.approx(0.0)


def test_clustered_delta_requires_shared_donors(pooled_preds):
    a, b, _ = pooled_preds
    b_disjoint = {s: d.rename(index={i: f"x{i}" for i in d.index})
                  for s, d in b.items()}
    res = inf.clustered_delta_rho(a, b_disjoint, n_boot=50, seed=0)
    assert np.isnan(res["delta_rho"])
    assert res["n_boot_valid"] == 0


def test_mismatched_observed_raises():
    a = {0: pd.DataFrame({"observed": [1, 2, 3], "predicted": [1, 2, 3]},
                         index=["a", "b", "c"])}
    b = {0: pd.DataFrame({"observed": [3, 2, 1], "predicted": [1, 2, 3]},
                         index=["a", "b", "c"])}
    with pytest.raises(ValueError, match="observed"):
        inf.clustered_delta_rho(a, b, n_boot=10)


def test_benjamini_hochberg():
    p = pd.Series([0.001, 0.01, 0.04, 0.5, np.nan])
    q = inf.benjamini_hochberg(p)
    assert q.iloc[0] <= p.iloc[0] * 4 + 1e-12
    assert q.iloc[3] > q.iloc[0]
    assert np.isnan(q.iloc[4])
    assert (q.dropna() <= 1).all()


def test_load_perm_nulls(tmp_path):
    d = tmp_path / "perm_null_s0"
    d.mkdir()
    for k in range(3):
        pd.DataFrame({"arm": ["B2", "T2"], "perm": [k, k],
                      "rho": [0.1 * k, -0.1 * k],
                      "n_test": [20, 20]}).to_csv(d / f"perm_{k}.csv",
                                                  index=False)
    df = inf.load_perm_nulls(d)
    assert len(df) == 6
    q95, n = inf.null_quantile(df, "B2")
    assert n == 3
    assert q95 == pytest.approx(df[df.arm == "B2"].rho.quantile(0.95))
    q, n0 = inf.null_quantile(df, "T9")
    assert n0 == 0 and np.isnan(q)


def _comp_row(delta, lo):
    return pd.DataFrame({"delta_rho": [delta], "ci_low": [lo]},
                        index=["T2-B2"])


def test_gates_improvement_and_null():
    comp = pd.DataFrame(
        {"delta_rho": [0.08, 0.06], "ci_low": [0.02, 0.01]},
        index=["T2-B2", "T2-T1"])
    nulls = pd.DataFrame({"arm": ["T2"] * 100,
                          "rho": np.linspace(-0.1, 0.2, 100)})
    g = inf.evaluate_gates(
        integrity_ok=True, comparisons=comp,
        arm_rhos={"T2": 0.55}, nulls=nulls,
        transfer_rhos={"a_to_b": [0.4, 0.5], "b_to_a": [0.3]},
        composition={"delta_rho": 0.1, "ci_low": 0.01},
        matrix_complete=True, candidates=("T2",))
    assert g["integrity"] == "PASS"
    assert g["improvement_T2_vs_B2"] == "PASS"
    assert g["semantics_T2_vs_T1"] == "PASS"
    assert g["semantics_T2_null"] == "PASS"
    assert g["historical_G5b_T2"] == "PASS"
    assert g["transfer"]["a_to_b"] == "PASS"
    assert g["transfer"]["b_to_a"] == "FAIL"
    assert g["composition"] == "PASS"
    assert g["negative_result_valid"] == "PASS"


def test_gates_fail_when_below_null():
    comp = pd.DataFrame(
        {"delta_rho": [0.08, 0.06], "ci_low": [0.02, 0.01]},
        index=["T2-B2", "T2-T1"])
    nulls = pd.DataFrame({"arm": ["T2"] * 100,
                          "rho": np.linspace(0.5, 0.6, 100)})
    g = inf.evaluate_gates(
        integrity_ok=True, comparisons=comp,
        arm_rhos={"T2": 0.4}, nulls=nulls,
        transfer_rhos={}, composition=None,
        matrix_complete=False, candidates=("T2",))
    assert g["improvement_T2_vs_B2"] == "FAIL"
    assert g["semantics_T2_null"] == "FAIL"
    assert g["composition"] == "NOT_EVALUABLE"
    assert g["negative_result_valid"] == "FAIL"


def test_gates_not_evaluable_without_integrity():
    comp = pd.DataFrame({"delta_rho": [0.2], "ci_low": [0.1]},
                        index=["T2-B2"])
    g = inf.evaluate_gates(
        integrity_ok=False, comparisons=comp,
        arm_rhos={"T2": 0.9}, nulls=pd.DataFrame(columns=["arm", "rho"]),
        transfer_rhos={}, composition=None,
        matrix_complete=True, candidates=("T2",))
    assert g["integrity"] == "NOT_EVALUABLE"
    assert g["improvement_T2_vs_B2"] == "NOT_EVALUABLE"
    assert g["semantics_T2_null"] == "NOT_EVALUABLE"
