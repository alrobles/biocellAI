"""M7 — donor-level progression evaluation (gates G5a/G5b).

Cell embeddings → donor embeddings (mean pool) → ridge regression on a
continuous pathology axis (default: Allen CPS_Global). Baseline: donor
pseudobulk PCA. All evaluation is held-out-donor: regression is fit on
train donors only, ρ/R² computed on held-out donors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BRAAK_ORDER = {"Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3,
               "Braak IV": 4, "Braak V": 5, "Braak VI": 6}
CERAD_ORDER = {"Absent": 0, "Sparse": 1, "Moderate": 2, "Frequent": 3}
ADNC_ORDER = {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3}


def donor_embeddings(cell_emb: np.ndarray, donor_ids) -> pd.DataFrame:
    """Mean-pool cell embeddings per donor → donor embedding matrix."""
    df = pd.DataFrame(cell_emb)
    df["donor"] = np.asarray(donor_ids)
    return df.groupby("donor").mean()


def pseudobulk_donor(X, donor_ids, n_components: int = 20, seed: int = 0,
                     train_donors=None):
    """Expression-only baseline: mean expression per donor → PCA."""
    from sklearn.decomposition import PCA

    if train_donors is None:
        raise ValueError("train_donors is required for inductive pseudobulk PCA")
    pb = donor_embeddings(
        np.asarray(X.todense() if hasattr(X, "todense") else X), donor_ids)
    train = list(train_donors)
    if len(train) < 2 or not set(train).issubset(pb.index):
        raise ValueError("PCA needs at least two valid training donors")
    n = min(n_components, len(train) - 1, pb.shape[1])
    pca = PCA(n_components=n, random_state=seed).fit(pb.loc[train])
    return pd.DataFrame(pca.transform(pb), index=pb.index)


def regress_heldout(donor_emb: pd.DataFrame, target: pd.Series,
                    train_donors, test_donors, alpha: float = 1.0):
    """Ridge on train donors → predict held-out → Spearman ρ, R²."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    tr, te = _evaluation_donors(donor_emb, target, train_donors, test_donors)
    if len(tr) < 5 or len(te) < 3:
        return {"rho": np.nan, "r2": np.nan, "mae": np.nan,
                "n_train": len(tr), "n_test": len(te)}
    pred = ridge_predictions(donor_emb, target, tr, te, alpha)
    rho = spearmanr(pred.predicted, pred.observed).statistic
    return {"rho": float(rho), "r2": float(r2_score(pred.observed, pred.predicted)),
            "mae": float(mean_absolute_error(pred.observed, pred.predicted)),
            "n_train": len(tr), "n_test": len(te)}


def _evaluation_donors(features, target, train_donors, test_donors):
    tr, te = list(train_donors), list(test_donors)
    if not features.index.is_unique or not target.index.is_unique:
        raise ValueError("donor indices must be unique")
    if len(set(tr)) != len(tr) or len(set(te)) != len(te):
        raise ValueError("duplicate donors in split")
    if set(tr) & set(te):
        raise ValueError("train/test donor overlap")
    tr = [d for d in tr if d in features.index and pd.notna(target.get(d))]
    te = [d for d in te if d in features.index and pd.notna(target.get(d))]
    return tr, te


def ridge_predictions(donor_emb, target, train_donors, test_donors, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, te = _evaluation_donors(donor_emb, target, train_donors, test_donors)
    if len(tr) < 5 or len(te) < 3:
        raise ValueError("ridge requires at least 5 train and 3 test donors")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(donor_emb.loc[tr].to_numpy(), target.loc[tr].to_numpy())
    return pd.DataFrame({"observed": target.loc[te].to_numpy(),
                         "predicted": model.predict(donor_emb.loc[te].to_numpy())},
                        index=pd.Index(te, name="donor_id"))


def trajectory_correlation(donor_emb: pd.DataFrame, ordinal: pd.Series,
                           train_donors: set | None = None):
    """PC1 of donor embeddings as pseudo-progression order vs an ordinal
    pathology score (Braak/CERAD/ADNC). Spearman over donors with labels.

    PCA sign is arbitrary: when `train_donors` is given, the PC1 direction
    is oriented to correlate positively with the ordinal on TRAIN donors
    and the reported rho is computed on the remaining (held-out) donors —
    an honest trajectory test. Without train_donors the raw (sign-
    ambiguous) rho over all donors is returned.
    """
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA

    if not donor_emb.index.is_unique or not ordinal.index.is_unique:
        raise ValueError("donor indices must be unique")
    common = [d for d in donor_emb.index if pd.notna(ordinal.get(d))]
    tr = common if train_donors is None else [d for d in common if d in train_donors]
    te = common if train_donors is None else [d for d in common if d not in train_donors]
    protocol = "descriptive_all_donors" if train_donors is None else "inductive_v2"
    if len(tr) < 3 or len(te) < 3:
        return {"rho": np.nan, "n": len(te), "protocol": protocol}
    pca = PCA(n_components=1, random_state=0).fit(donor_emb.loc[tr].values)
    pc1 = pd.Series(pca.transform(donor_emb.loc[common].values).ravel(), index=common)
    if train_donors is not None:
        r_tr = spearmanr(pc1[tr], ordinal[tr]).statistic
        if not np.isfinite(r_tr) or r_tr == 0:
            return {"rho": np.nan, "n": len(te), "protocol": protocol}
        if r_tr < 0:
            pc1 = -pc1
    rho, _ = spearmanr(pc1[te], ordinal[te])
    return {"rho": float(rho), "n": len(te), "protocol": protocol}


def paired_rho_difference(a, b, n_boot: int = 2000, seed: int = 0):
    from scipy.stats import spearmanr

    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
        raise ValueError("paired comparisons require the same unique donors")
    b = b.loc[a.index]
    if not np.array_equal(a.observed.to_numpy(), b.observed.to_numpy()):
        raise ValueError("paired observed targets must match")
    values = np.column_stack([a.observed, a.predicted, b.predicted]).astype(float)
    if len(values) < 3 or not np.isfinite(values).all() or n_boot < 1:
        raise ValueError("need finite predictions, >=3 donors and positive n_boot")
    if any(np.ptp(v) == 0 for v in values.T):
        raise ValueError("correlations require nonconstant targets and predictions")
    delta = float(spearmanr(values[:, 0], values[:, 1]).statistic
                  - spearmanr(values[:, 0], values[:, 2]).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        if any(np.ptp(v) == 0 for v in sample.T):
            continue
        boots.append(spearmanr(sample[:, 0], sample[:, 1]).statistic
                     - spearmanr(sample[:, 0], sample[:, 2]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"delta_rho": delta, "ci_low": float(lo), "ci_high": float(hi),
            "n": len(values), "n_boot_valid": len(boots)}


def ordinal_map(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).astype(float)
