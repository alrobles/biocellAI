import numpy as np
import anndata
import pandas as pd


def make_adata():
    """Synthetic 60-cell x 30-gene AnnData with 3 blocky cell types, 2 donors."""
    rng = np.random.default_rng(0)
    X = rng.poisson(0.2, (60, 30)).astype(np.float32)
    labels = np.repeat(["T", "B", "Mono"], 20)
    for i, block in enumerate([range(0, 10), range(10, 20), range(20, 30)]):
        X[np.ix_(labels == np.array(["T", "B", "Mono"])[i], list(block))] += 5.0
    obs = pd.DataFrame({"cell_type": labels, "donor_id": np.tile(["d0", "d1"], 30)})
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(30)])
    return anndata.AnnData(X=X, obs=obs, var=var)
