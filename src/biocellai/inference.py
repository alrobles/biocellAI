"""R6-INFERENCE — paired donor-level comparisons, permutation nulls, gates.

ADR-011 rules implemented here:
- A donor, not a cell or a seed, is the independent evaluation unit.
- Paired donor-level bootstrap (>=2000 valid replicates); when folds are
  repeated across seeds, donor identity — not seed x donor pairs — is the
  resampling cluster.
- The Improvement gate requires an exact delta_rho >= 0.05 over the paired
  v2 reference AND a lower paired CI bound > 0 (computed without rounding).
- Secondary comparisons get BH-FDR on the empirical bootstrap two-sided
  p-value; nonsignificance is not equivalence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2)


def load_predictions(path: Path, arm: str) -> pd.DataFrame:
    """predictions_<arm>.csv -> df indexed by donor_id with observed/predicted."""
    f = Path(path) / f"predictions_{arm}.csv"
    df = pd.read_csv(f)
    df = df.rename(columns={df.columns[0]: "donor_id"})
    return df.set_index("donor_id")[["observed", "predicted"]]


def _rho(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _seed_delta(pred_a, pred_b) -> tuple[float, pd.DataFrame]:
    """Paired delta-rho on shared donors of one seed; returns (delta, merged)."""
    m = pred_a.join(pred_b, lsuffix="_a", rsuffix="_b", how="inner")
    m = m.dropna()
    if len(m) < 3 or not np.isfinite(m.to_numpy()).all():
        return np.nan, m
    if np.ptp(m.observed_a.to_numpy()) == 0:
        return np.nan, m
    if not np.allclose(m.observed_a, m.observed_b):
        raise ValueError("paired observed targets differ between arms")
    return _rho(m.observed_a, m.predicted_a) - _rho(
        m.observed_b, m.predicted_b), m


def clustered_delta_rho(preds_a: dict[int, pd.DataFrame],
                        preds_b: dict[int, pd.DataFrame],
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired delta-rho pooled across seeds with donor-clustered bootstrap.

    preds_*: {seed: predictions df}. The point estimate is the mean over
    seeds of the per-seed paired delta. The bootstrap resamples DONOR
    identities (all observations of a donor across seeds move together),
    so repeated seed folds never count one donor as independent replicates.
    Returns delta_rho, ci_low, ci_high, n_boot_valid, per-seed deltas.
    """
    seeds = sorted(set(preds_a) & set(preds_b))
    merged = {}
    point = {}
    for s in seeds:
        d, m = _seed_delta(preds_a[s], preds_b[s])
        if m.empty:
            continue
        merged[s] = m
        point[s] = d
    if not merged:
        return {"delta_rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_boot_valid": 0, "per_seed": {}, "n_donors": 0}
    point_val = float(np.nanmean(list(point.values())))

    # pooled row set: (seed, donor) — donors shared across seeds are the
    # resampling units; each donor contributes its available seed rows.
    # Flat numpy arrays: bootstrap iterations only gather indices.
    seed_arr, obs_arr, pa_arr, pb_arr, donor_arr = [], [], [], [], []
    for s, m in merged.items():
        n = len(m)
        seed_arr.append(np.full(n, s))
        obs_arr.append(m.observed_a.to_numpy(dtype=float))
        pa_arr.append(m.predicted_a.to_numpy(dtype=float))
        pb_arr.append(m.predicted_b.to_numpy(dtype=float))
        donor_arr.append(m.index.to_numpy())
    seed_v = np.concatenate(seed_arr)
    obs_v = np.concatenate(obs_arr)
    pa_v = np.concatenate(pa_arr)
    pb_v = np.concatenate(pb_arr)
    donor_v = np.concatenate(donor_arr)
    donors = np.unique(donor_v)
    rows_of_donor = {d: np.flatnonzero(donor_v == d) for d in donors}
    seed_ids = np.unique(seed_v)

    def stat(sampled):
        idx = np.concatenate([rows_of_donor[d] for d in sampled
                              if d in rows_of_donor])
        deltas = []
        for s in seed_ids:
            sel = idx[seed_v[idx] == s]
            if len(sel) < 3 or np.ptp(obs_v[sel]) == 0:
                continue
            deltas.append(_rho(obs_v[sel], pa_v[sel])
                          - _rho(obs_v[sel], pb_v[sel]))
        return float(np.mean(deltas)) if deltas else np.nan

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = donors[rng.integers(0, len(donors), len(donors))]
        v = stat(sample)
        if np.isfinite(v):
            boots.append(v)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) >= 20
              else (np.nan, np.nan))
    # empirical two-sided bootstrap p: mass on the opposite side of 0
    arr = np.asarray(boots)
    if len(arr):
        p_emp = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
        p_emp = float(min(1.0, p_emp + 1.0 / (len(arr) + 1)))
    else:
        p_emp = np.nan
    return {"delta_rho": point_val, "ci_low": float(lo),
            "ci_high": float(hi), "n_boot_valid": int(len(arr)),
            "per_seed": {s: float(v) for s, v in point.items()},
            "n_donors": int(len(donors)), "p_emp": p_emp}


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """BH-FDR q-values for a set of p-values (NaN-safe)."""
    p = pvals.astype(float)
    out = pd.Series(np.nan, index=p.index)
    ok = p.dropna()
    if ok.empty:
        return out
    order = ok.sort_values()
    m = len(order)
    q = order * m / np.arange(1, m + 1)
    q = q.iloc[::-1].cummin().iloc[::-1]
    out.loc[order.index] = np.minimum(q, 1.0)
    return out


def load_perm_nulls(perm_dir: Path) -> pd.DataFrame:
    """Read perm_<k>.csv rows -> DataFrame[arm, perm, rho, n_test]."""
    rows = []
    for f in sorted(Path(perm_dir).glob("perm_*.csv")):
        rows.append(pd.read_csv(f))
    if not rows:
        return pd.DataFrame(columns=["arm", "perm", "rho", "n_test"])
    return pd.concat(rows, ignore_index=True)


def null_quantile(null: pd.DataFrame, arm: str, level: float = 0.95):
    """Null distribution quantile for an arm (pooled across seeds)."""
    sub = null.loc[null.arm == arm, "rho"].dropna()
    if sub.empty:
        return np.nan, 0
    return float(sub.quantile(level)), int(len(sub))


def evaluate_gates(integrity_ok, comparisons: pd.DataFrame,
                   arm_rhos: dict, nulls: pd.DataFrame,
                   transfer_rhos: dict, composition: dict | None,
                   matrix_complete: bool,
                   candidates=("T2", "T3"), threshold=0.05,
                   g5b_min=0.5, transfer_min=0.35) -> dict:
    """Automated ADR-011 gate decisions.

    comparisons: df from clustered_delta_rho indexed by "arm-ref" (e.g.
      "T2-B2"), primary cohort only.
    arm_rhos: {arm: mean rho} on the primary cohort (registered endpoint).
    nulls: perm-null df (load_perm_nulls); null_ok = arm rho > null q95.
    transfer_rhos: {direction: [strict rho per seed]}.
    composition: {"delta_rho":, "ci_low":} of the state block over cov+comp.
    """
    gates, notes = {}, []
    gates["integrity"] = "PASS" if integrity_ok else "NOT_EVALUABLE"
    if not integrity_ok:
        notes.append("integrity unverifiable -> downstream gates blocked")

    def delta_of(a, ref):
        key = f"{a}-{ref}"
        return (comparisons.loc[key] if key in comparisons.index else None)

    for cand in candidates:
        rho_c = arm_rhos.get(cand, np.nan)
        q95, n_null = null_quantile(nulls, cand)
        null_ok = bool(np.isfinite(rho_c) and np.isfinite(q95)
                       and rho_c > q95)
        for ref, tag in (("B2", "improvement"), ("T1", "semantics")):
            d = delta_of(cand, ref)
            name = f"{tag}_{cand}_vs_{ref}"
            if d is None or not np.isfinite(d["delta_rho"]):
                gates[name] = "NOT_EVALUABLE"
            elif not integrity_ok:
                gates[name] = "NOT_EVALUABLE"
            elif not null_ok:
                gates[name] = "FAIL" if tag == "improvement" else \
                    "FAIL"  # semantics requires beating the null
                notes.append(f"{name}: rho {rho_c:.3f} <= null q95 "
                             f"{q95:.3f} (n_null={n_null})")
            else:
                gates[name] = ("PASS" if d["delta_rho"] >= threshold
                               and d["ci_low"] > 0 else "FAIL")
        gates[f"semantics_{cand}_null"] = (
            "PASS" if null_ok else
            "FAIL" if np.isfinite(q95) else "NOT_EVALUABLE")
        gates[f"historical_G5b_{cand}"] = (
            "PASS" if np.isfinite(rho_c) and rho_c >= g5b_min else "FAIL")

    gates["transfer"] = {
        d: ("PASS" if vals and min(vals) >= transfer_min else
            "FAIL" if vals else "NOT_EVALUABLE")
        for d, rhos in transfer_rhos.items()
        for vals in [[r for r in rhos if np.isfinite(r)]]}

    if composition and np.isfinite(composition.get("delta_rho", np.nan)):
        gates["composition"] = (
            "PASS" if composition["delta_rho"] > 0
            and composition.get("ci_low", 0) > 0 else "FAIL")
        notes.append("composition: no automatic causal attribution")
    else:
        gates["composition"] = "NOT_EVALUABLE"

    gates["negative_result_valid"] = (
        "PASS" if matrix_complete else "FAIL")
    gates["notes"] = notes
    return gates
