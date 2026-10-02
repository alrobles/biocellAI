import numpy as np
import pandas as pd

from biocellai.progression import (BRAAK_ORDER, donor_embeddings, ordinal_map,
                               regress_heldout, trajectory_correlation)


def _toy(n_donors=20, cells_per=30, d=8, seed=0):
    rng = np.random.default_rng(seed)
    donor_vec = rng.normal(size=(n_donors, d))
    donors = np.repeat([f"d{i}" for i in range(n_donors)], cells_per)
    emb = donor_vec[np.repeat(np.arange(n_donors), cells_per)] + 0.1 * rng.normal(
        size=(n_donors * cells_per, d))
    return emb, donors, pd.Series(donor_vec[:, 0], index=[f"d{i}" for i in range(n_donors)])


def test_donor_embeddings_pools_per_donor():
    emb, donors, _ = _toy()
    pooled = donor_embeddings(emb, donors)
    assert pooled.shape == (20, 8)
    assert set(pooled.index) == {f"d{i}" for i in range(20)}


def test_regress_heldout_recovers_signal():
    emb, donors, target = _toy()
    pooled = donor_embeddings(emb, donors)
    train = [f"d{i}" for i in range(15)]
    test = [f"d{i}" for i in range(15, 20)]
    res = regress_heldout(pooled, target, train, test)
    assert res["n_test"] == 5
    assert res["rho"] > 0.5  # embedded direction is the target


def test_trajectory_correlation():
    rng = np.random.default_rng(0)
    prog = np.linspace(-2, 2, 24)
    emb = pd.DataFrame(
        prog[:, None] + 0.05 * rng.normal(size=(24, 4)),
        index=[f"d{i}" for i in range(24)])
    braak = pd.Series({f"d{i}": f"Braak {['0','I','II','III','IV','V'][int(i/4)]}"
                       for i in range(24)})
    res = trajectory_correlation(emb, ordinal_map(braak, BRAAK_ORDER))
    assert abs(res["rho"]) > 0.9
    assert res["n"] == 24
