import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from sklearn.decomposition import PCA

from biocellai.data import donor_split, prepare_donor_fold, preprocess, split_donor_ids
from biocellai.progression import (
    paired_rho_difference,
    pseudobulk_donor,
    regress_heldout,
    ridge_predictions,
    trajectory_correlation,
)
from biocellai.revalidation import improvement_gate, plan_status, validate_plan


def counts():
    rng = np.random.default_rng(14)
    return AnnData(
        X=rng.poisson(3.0, (80, 400)).astype(np.float32),
        obs=pd.DataFrame(
            {"donor_id": np.repeat([f"d{i}" for i in range(8)], 10)},
            index=[f"c{i}" for i in range(80)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(400)]),
    )


def test_repeated_preprocess_rejected():
    first = preprocess(counts(), n_hvg=300)
    with pytest.raises(ValueError, match="counts|processed"):
        preprocess(first, n_hvg=300)


@pytest.mark.parametrize("n", [9, 84, 111, 127])
def test_single_split_rule(n):
    ids = np.array([f"d{i}" for i in range(n)])
    a = AnnData(X=np.ones((n, 2)), obs=pd.DataFrame({"donor_id": ids}))
    train, test = split_donor_ids(ids, seed=2)
    tr_mask, te_mask = donor_split(a, seed=2)
    assert len(test) == max(1, round(n * 0.25))
    assert set(ids[tr_mask]) == set(train)
    assert set(ids[te_mask]) == set(test)
    assert set(train).isdisjoint(test)
    assert np.array_equal(test, split_donor_ids(ids[::-1], seed=2)[1])


def test_fold_gene_selection_and_train_values_ignore_test():
    a = counts()
    train, test = split_donor_ids(a.obs.donor_id, seed=0)
    one = prepare_donor_fold(a, train, test, n_hvg=100)
    a.X[a.obs.donor_id.isin(test)] *= np.arange(1, 401)
    two = prepare_donor_fold(a, train, test, n_hvg=100)
    assert one.var_names.equals(two.var_names)
    tr = one.obs.donor_id.isin(train)
    np.testing.assert_allclose(one[tr].X, two[tr].X)
    assert one.uns["biocellai_protocol"] == "v2"
    assert one.uns["biocellai_data_state"] == "log1p_hvg"
    assert one.obs["is_test"].sum() == 20


def test_fold_rejects_bad_counts_and_split():
    a = counts()
    train, test = split_donor_ids(a.obs.donor_id)
    with pytest.raises(ValueError, match="overlap"):
        prepare_donor_fold(a, train, [*test, train[0]])
    a.X[0, 0] = 0.5
    with pytest.raises(ValueError, match="counts"):
        prepare_donor_fold(a, train, test)


def test_fold_rejects_preprocessed_input_and_duplicate_cells():
    a = counts()
    train, test = split_donor_ids(a.obs.donor_id)
    a.uns["log1p"] = {"base": None}
    with pytest.raises(ValueError, match="counts|processed"):
        prepare_donor_fold(a, train, test)
    del a.uns["log1p"]
    a.obs_names = ["duplicate"] * a.n_obs
    with pytest.raises(ValueError, match="unique"):
        prepare_donor_fold(a, train, test)


def test_pca_fits_training_donors_only(monkeypatch):
    seen = []
    fit, fit_transform = PCA.fit, PCA.fit_transform

    def record_fit(self, X, *args, **kwargs):
        seen.append(len(X))
        return fit(self, X, *args, **kwargs)

    def record_fit_transform(self, X, *args, **kwargs):
        seen.append(len(X))
        return fit_transform(self, X, *args, **kwargs)

    monkeypatch.setattr(PCA, "fit", record_fit)
    monkeypatch.setattr(PCA, "fit_transform", record_fit_transform)
    rng = np.random.default_rng(0)
    ids = [f"d{i}" for i in range(12)]
    X = rng.normal(size=(12, 5))
    emb = pd.DataFrame(X, index=ids)
    y = pd.Series(np.arange(12), index=ids)
    trajectory_correlation(emb, y, train_donors=set(ids[:9]))
    pseudobulk_donor(X, ids, train_donors=ids[:9])
    assert seen and set(seen) == {9}


def test_pseudobulk_requires_training_split():
    with pytest.raises(ValueError, match="train_donors"):
        pseudobulk_donor(np.ones((8, 3)), [f"d{i}" for i in range(8)])


def test_ridge_invariant_to_feature_units_and_exports_predictions():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(20, 4)), index=[f"d{i}" for i in range(20)])
    y = pd.Series(X[0] - X[1], index=X.index)
    train, test = X.index[:15], X.index[15:]
    a = ridge_predictions(X, y, train, test)
    b = ridge_predictions(X * [1e6, 1e-4, 5, 1], y, train, test)
    np.testing.assert_allclose(a.predicted, b.predicted, atol=1e-8)
    assert list(a.index) == list(test)
    assert set(a.columns) == {"observed", "predicted"}
    r = regress_heldout(X, y, train, test)
    assert r["n_test"] == 5 and r["rho"] > 0.8
    assert "mae" in r


def test_regression_rejects_overlapping_donors():
    X = pd.DataFrame(np.ones((8, 2)), index=[f"d{i}" for i in range(8)])
    y = pd.Series(np.arange(8), index=X.index)
    with pytest.raises(ValueError, match="overlap"):
        regress_heldout(X, y, X.index[:6], X.index[5:])


def test_paired_bootstrap_and_alignment():
    y = np.arange(12, dtype=float)
    a = pd.DataFrame({"observed": y, "predicted": y}, index=[f"d{i}" for i in range(12)])
    b = a.assign(predicted=-y)
    r = paired_rho_difference(a, b.iloc[::-1], n_boot=100, seed=3)
    assert r["delta_rho"] == pytest.approx(2.0)
    assert r["ci_low"] == pytest.approx(2.0)
    assert r["n"] == 12 and r["n_boot_valid"] > 90
    assert r == paired_rho_difference(a, b, n_boot=100, seed=3)
    with pytest.raises(ValueError, match="donors"):
        paired_rho_difference(a, b.iloc[:-1])
    with pytest.raises(ValueError, match="observed"):
        paired_rho_difference(a, b.assign(observed=y + 1))


def test_gates_never_round_or_pass_missing_controls():
    assert improvement_gate(0.604, 0.558, 0.01, True, True) == "FAIL"
    assert improvement_gate(0.650, 0.600, 0.01, True, True) == "PASS"
    assert improvement_gate(0.7, 0.6, -0.01, True, True) == "FAIL"
    assert improvement_gate(0.7, 0.6, 0.01, True, None) == "NOT_EVALUABLE"
    assert improvement_gate(0.7, 0.6, 0.01, False, True) == "NOT_EVALUABLE"


def test_plan_dependencies_and_no_automatic_completion():
    plan = {"tasks": [
        {"id": "A", "status": "completed", "depends_on": [], "evidence": ["test log"]},
        {"id": "B", "status": "pending", "depends_on": ["A"]},
        {"id": "C", "status": "pending", "depends_on": ["B"]},
    ]}
    validate_plan(plan)
    assert plan_status(plan) == {"A": "completed", "B": "ready", "C": "blocked"}
    plan["tasks"][0]["depends_on"] = ["C"]
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_plan_rejects_false_completion():
    with pytest.raises(ValueError, match="evidence"):
        validate_plan({"tasks": [{"id": "A", "status": "completed", "depends_on": []}]})


def test_committed_task_registry_is_valid():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "spec/revalidation_tasks.json"
    plan = json.loads(path.read_text())
    validate_plan(plan)
    assert plan_status(plan)["R9-M13"] == "blocked"


@pytest.mark.parametrize("script", ["m7_progression.py", "m11_rosmap_eval.py"])
def test_readout_baseline_uses_same_donor_count(tmp_path, script):
    import subprocess
    import sys
    from pathlib import Path

    rng = np.random.default_rng(2)
    ids = np.array([f"d{i}" for i in range(20)])
    donor = np.repeat(ids, 4)
    X = rng.normal(size=(len(donor), 6))
    a = AnnData(X=X, obs=pd.DataFrame({"donor_id": donor}))
    h5ad = tmp_path / "legacy.h5ad"
    a.write_h5ad(h5ad)
    _, test = split_donor_ids(donor, seed=0)
    embdir = tmp_path / "embeddings"
    embdir.mkdir()
    np.savez(embdir / "emb_s0_control_all-MiniLM-L6-v2.npz", emb=X,
             donor=donor, cell_type=np.repeat("type", len(donor)),
             region=np.repeat("MTG", len(donor)), is_test=np.isin(donor, test))
    donor_table = tmp_path / "donors.csv"
    pd.DataFrame({"CPS_Global": rng.normal(size=20), "braak": ["Braak III"] * 10 + ["Braak VI"] * 10,
                  "cerad": ["Sparse"] * 10 + ["Frequent"] * 10,
                  "adnc": ["Low"] * 10 + ["High"] * 10,
                  "pathology": ["earlyAD"] * 10 + ["lateAD"] * 10,
                  "path_level": [1] * 10 + [2] * 10, "region": ["MTG"] * 20},
                 index=pd.Index(ids, name="donor_id")).to_csv(donor_table)
    out = tmp_path / "out"
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts" / script), "--emb-dir", str(embdir),
               "--donor-table", str(donor_table), "--h5ad", str(h5ad), "--seeds", "0", "--out", str(out)]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    metrics = pd.read_csv(out / "progression_metrics.csv")
    regression = metrics[metrics.n_test.notna()]
    assert set(regression.n_test) == {len(test)}
    assert set(metrics.evaluation_protocol) == {"v2_readout_on_unrevalidated_inputs"}


def test_saved_mask_cannot_split_a_donor():
    from biocellai.data import donors_from_mask

    with pytest.raises(ValueError, match="overlap"):
        donors_from_mask(["a", "a", "b", "b"], [0, 1, 0, 1])
    tr, te = donors_from_mask(["a", "a", "b", "b"], [0, 0, 1, 1])
    assert tr.tolist() == ["a"] and te.tolist() == ["b"]


def test_v2_prepare_cli_is_immutable(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    import anndata as ad

    from biocellai.revalidation import file_sha256

    raw = tmp_path / "raw.h5ad"
    counts().write_h5ad(raw)
    out = tmp_path / "prepared"
    script = Path(__file__).resolve().parents[1] / "scripts/v2_prepare.py"
    command = [sys.executable, str(script), "--input", str(raw),
               "--counts-layer", "X", "--cohort", "synthetic", "--source-release", "fixture",
               "--seeds", "0", "--n-hvg", "100", "--out", str(out)]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((out / "fold_s0.json").read_text())
    assert manifest["source"]["sha256"] == file_sha256(raw)
    matrix_hash = file_sha256(out / "fold_s0.h5ad")
    assert manifest["matrix_sha256"] == matrix_hash
    a = ad.read_h5ad(out / "fold_s0.h5ad")
    assert a.uns["biocellai_protocol"] == "v2"
    assert a.n_vars == 100
    assert json.loads((out / "PREPARED.json").read_text())["scientific_results_complete"] is False
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0
    assert file_sha256(out / "fold_s0.h5ad") == matrix_hash


def test_label_prototypes_are_fixed_and_semantics_free():
    from biocellai.train import class_prototypes

    labels = ["T", "B", "T"]
    cells, bank, cats = class_prototypes(labels, mode="random", dim=16, seed=5)
    assert cats == ["B", "T"]
    assert cells.shape == (3, 16)
    np.testing.assert_array_equal(cells[0], cells[2])
    np.testing.assert_array_equal(bank, class_prototypes(labels, "random", 16, 5)[1])
    assert not np.array_equal(bank, class_prototypes(labels, "random", 16, 6)[1])
    _, onehot, _ = class_prototypes(labels, "onehot", 16, 0)
    np.testing.assert_array_equal(onehot.numpy() @ onehot.numpy().T, np.eye(2))
    with pytest.raises(ValueError):
        class_prototypes(labels, "onehot", 1, 0)


def test_shuffle_preserves_test_labels_and_train_marginals():
    from biocellai.data import shuffle_training_donor_labels

    obs = pd.DataFrame({"donor_id": np.repeat([f"d{i}" for i in range(8)], 2),
                        "path": np.repeat(np.arange(8), 2)})
    train = [f"d{i}" for i in range(6)]
    shuffled = shuffle_training_donor_labels(obs, "path", train, seed=6)
    is_train = obs.donor_id.isin(train)
    np.testing.assert_array_equal(shuffled[~is_train], obs.loc[~is_train, "path"])
    assert sorted(shuffled[is_train].unique()) == list(range(6))
    assert not np.array_equal(shuffled[is_train], obs.loc[is_train, "path"])
    assert (pd.DataFrame({"d": obs.donor_id, "y": shuffled}).groupby("d").y.nunique() == 1).all()


def test_sparse_fold_and_full_library_normalization():
    from scipy.sparse import csr_matrix

    a = counts()
    train, test = split_donor_ids(a.obs.donor_id)
    expected = np.log1p(a.X / a.X.sum(axis=1, keepdims=True) * 1e4)
    a.X = csr_matrix(a.X)
    fold = prepare_donor_fold(a, train, test, n_hvg=50)
    indices = a.var_names.get_indexer(fold.var_names)
    np.testing.assert_allclose(fold.X.toarray(), expected[:, indices], rtol=1e-6)
