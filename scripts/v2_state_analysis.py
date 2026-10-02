#!/usr/bin/env python3
"""R5-STATE — composition vs state for donor-level pathology.

For each fold seed of a pathology cohort, builds donor-level feature
blocks and evaluates ridge pathology prediction on held-out donors:

  cov       donor covariates (age, sex, apoe/PMI...)
  comp      donor x cell-type fractions (train vocab only)
  compXreg  donor x (cell-type x region) fractions (multi-region only)
  state_pb  pseudobulk -> train-fit PCA-20 (B1-style, inductive)
  state_scfm donor-mean of a frozen scFM embedding npz

Analyses (all train-fitted, fixed weights on test — no test fitting):
  1. blockwise ridge rho for every block combination
  2. train-only residualization: r = y - (cov+comp fit) then
     rho(state -> r) — does expression state explain what composition
     + covariates do not?
  3. paired incremental comparison: bootstrap donors ->
     CI on delta-rho of (cov+comp+state) over (cov+comp)
  4. confound link: Spearman between the stored N1 (shuffled-label)
     held-out predictions and the cov+comp predictions

Associations only — no causal attribution.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from biocellai.progression import (donor_embeddings, pseudobulk_donor,
                                   ridge_predictions)
from biocellai.revalidation import file_sha256, reserve_output
from biocellai.v2run import composition_features, donor_target

SEED_SET = (0, 1, 2)
COHORT_CFG = {
    "seaad_mtg": dict(path="CPS_Global", type_col="cell_type",
                      region_col="region",
                      covs=("age_at_death", "sex", "apoe")),
    "rosmap": dict(path="path_level", type_col="cell_type",
                   region_col="BrainRegion",
                   covs=("age_death", "Sex", "pmi", "n_regions")),
}


def _covariates(adata, covs) -> pd.DataFrame:
    """Donor-level covariate frame (numeric; categoricals 0/1-coded)."""
    out = {}
    for c in covs:
        s = adata.obs[c].astype(str)
        if c == "apoe":  # "3/4" -> e4 dosage
            out["apoe_e4"] = s.str.count("4").groupby(
                adata.obs["donor_id"].astype(str)).first()
            continue
        vals = pd.to_numeric(s, errors="coerce")
        if vals.notna().mean() < 0.9:  # categorical -> binary
            cats = sorted(s.unique())
            if len(cats) == 2:
                vals = (s == cats[-1]).astype(float)
        out[c] = vals.groupby(adata.obs["donor_id"].astype(str)).first()
    return pd.DataFrame(out)


def _comp_x_region(adata, type_col, region_col,
                   train_mask) -> pd.DataFrame | None:
    """donor x (type,region) fractions; None when single-region."""
    if adata.obs[region_col].astype(str).nunique() < 2:
        return None
    pair = (adata.obs[type_col].astype(str) + "@" +
            adata.obs[region_col].astype(str))
    vocab = sorted(pair[train_mask].unique())
    oh = pd.get_dummies(pair).reindex(columns=vocab, fill_value=0)
    return donor_embeddings(oh.to_numpy(dtype=np.float32),
                            adata.obs["donor_id"].astype(str))


def _git_rev():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _rho(df):
    from scipy.stats import spearmanr
    return float(spearmanr(df.predicted, df.observed).statistic)


def _paired_delta(pred_a, pred_b, n_boot: int, rng):
    """Bootstrap donors -> CI on rho(A)-rho(B) (paired)."""
    from scipy.stats import spearmanr
    m = pred_a.join(pred_b, lsuffix="_a", rsuffix="_b", how="inner")
    m = m.dropna(subset=["observed_a", "observed_b"])
    deltas = []
    idx = np.arange(len(m))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            d = (spearmanr(m.predicted_a.iloc[b],
                           m.observed_a.iloc[b]).statistic
                 - spearmanr(m.predicted_b.iloc[b],
                             m.observed_b.iloc[b]).statistic)
            deltas.append(d)
        except Exception:
            continue
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return float(np.nanmean(deltas)), float(lo), float(hi), len(deltas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True, choices=list(COHORT_CFG))
    p.add_argument("--fold-dir", type=Path, required=True)
    p.add_argument("--scfm-npz", type=Path, required=True,
                   help="gf_v2_316m_native npz per seed ({s} placeholder)")
    p.add_argument("--n1-pred", type=Path, required=True,
                   help="run_s{seed}_full/predictions_N1.csv ({s} placeholder)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    cfg = COHORT_CFG[args.cohort]
    out = reserve_output(args.out)
    t0 = time.time()
    rng = np.random.default_rng(0)
    rows, resid_rows, link_rows = [], [], []
    seed_meta = []

    for seed in SEED_SET:
        fold = args.fold_dir / f"fold_s{seed}.h5ad"
        a = ad.read_h5ad(fold)
        obs = a.obs
        train_mask = ~obs["is_test"].astype(bool)
        donor = obs["donor_id"].astype(str)
        is_te = obs.assign(d=donor).groupby("d")["is_test"].first(
        ).astype(bool)
        tr = sorted(is_te.index[~is_te])
        te = sorted(is_te.index[is_te])
        target = donor_target(obs, cfg["path"])

        cov = _covariates(a, cfg["covs"])
        comp = composition_features(a, cfg["type_col"],
                                    np.asarray(train_mask))
        cxr = _comp_x_region(a, cfg["type_col"], cfg["region_col"],
                             np.asarray(train_mask))
        state_pb = pseudobulk_donor(a.X, donor, train_donors=tr)

        npz_path = Path(str(args.scfm_npz).format(s=seed))
        z = np.load(npz_path, allow_pickle=True)
        emb = z["emb"]
        names = z["obs_names"].astype(str)
        emb_df = pd.DataFrame(emb, index=names)
        emb_df["donor"] = obs.loc[emb_df.index, "donor_id"].astype(
            str).to_numpy()
        state_scfm = emb_df.groupby("donor").mean()

        blocks = {"cov": cov, "comp": comp,
                  "compXreg": cxr, "state_pb": state_pb,
                  "state_scfm": state_scfm}

        def join(*names):
            parts = [blocks[n] for n in names if blocks.get(n) is not None]
            if not parts:
                return None
            x = parts[0]
            for q in parts[1:]:
                x = x.join(q, how="outer", rsuffix="_r")
            x = x.fillna(0.0)
            x.columns = x.columns.astype(str)
            return x

        preds = {}
        combos = [("cov",), ("comp",), ("compXreg",), ("cov", "comp"),
                  ("cov", "comp", "compXreg"), ("state_pb",),
                  ("state_scfm",), ("cov", "comp", "state_pb"),
                  ("cov", "comp", "state_scfm")]
        blocks_run = []
        for names_t in combos:
            feats = join(*names_t)
            if feats is None:
                continue
            pr = ridge_predictions(feats, target, tr, te)
            name = "+".join(names_t)
            preds[name] = pr
            blocks_run.append(name)
            rows.append(dict(cohort=args.cohort, seed=seed, block=name,
                             rho=_rho(pr), n_feat=feats.shape[1],
                             n_train_donors=len(tr),
                             n_test=len(pr)))
        seed_meta.append(dict(
            seed=seed, n_train_donors=len(tr), n_test_donors=len(te),
            n_regions=int(obs[cfg["region_col"]].astype(str).nunique()),
            blocks_evaluated=blocks_run,
            compxreg_included=cxr is not None))

        # residualization: r = y - pred(cov+comp) on TRAIN (same
        # StandardScaler+Ridge(1.0) pipeline as ridge_predictions),
        # then state -> r on held-out donors
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        base_feats = join("cov", "comp")
        base_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        base_model.fit(base_feats.loc[tr], target.loc[tr])
        r_tr = target.loc[tr] - pd.Series(
            base_model.predict(base_feats.loc[tr]), index=tr)
        base_te = preds["cov+comp"]
        r_te = pd.Series(base_te.observed - base_te.predicted,
                         index=base_te.index)
        for sname in ("state_pb", "state_scfm"):
            feats = blocks[sname].reindex(target.index).fillna(0.0)
            r_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            r_model.fit(feats.loc[tr], r_tr)
            rpred = pd.Series(r_model.predict(feats.loc[te]), index=te)
            from scipy.stats import spearmanr
            rho_res = float(spearmanr(rpred, r_te.reindex(te)
                                      ).statistic)
            resid_rows.append(dict(cohort=args.cohort, seed=seed,
                                   state=sname,
                                   rho_residual=rho_res,
                                   n_test=len(rpred)))
            # paired incremental: full vs base on same donors
            d = _paired_delta(preds[f"cov+comp+{sname}"], base_te,
                              args.n_boot, rng)
            resid_rows[-1].update(
                dict(delta_rho=d[0], ci_lo=d[1], ci_hi=d[2],
                     n_boot=d[3]))

        # N1-confound link
        n1f = Path(str(args.n1_pred).format(s=seed))
        if n1f.exists():
            n1 = pd.read_csv(n1f, index_col=0)
            j = base_te.join(n1, how="inner", lsuffix="_cc",
                             rsuffix="_n1").dropna()
            rho_n1 = float(pd.Series(j.predicted_cc).corr(
                j.predicted_n1, method="spearman"))
            link_rows.append(dict(cohort=args.cohort, seed=seed,
                                  rho_n1_vs_covcomp=rho_n1,
                                  n_test=len(j)))

    res = {"step": "v2_state_analysis", "cohort": args.cohort,
           "pathology": cfg["path"], "covariates": list(cfg["covs"]),
           "seed_meta": seed_meta,
           "scfm_npz": str(args.scfm_npz),
           "scfm_npz_s0_sha256": file_sha256(
               Path(str(args.scfm_npz).format(s=0))),
           "n1_pred": str(args.n1_pred),
           "fold_s0_sha256": file_sha256(args.fold_dir / "fold_s0.h5ad"),
           "git": _git_rev(),
           "wall_s": round(time.time() - t0, 1)}
    (out / "blocks.csv").write_text(pd.DataFrame(rows).to_csv(index=False))
    (out / "residuals.csv").write_text(
        pd.DataFrame(resid_rows).to_csv(index=False))
    (out / "n1_link.csv").write_text(
        pd.DataFrame(link_rows).to_csv(index=False))
    (out / "manifest.json").write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(rows).pivot_table(index="block", columns="seed",
                                         values="rho").round(3).to_string())
    print(pd.DataFrame(resid_rows).round(3).to_string())
    print(pd.DataFrame(link_rows).round(3).to_string())


if __name__ == "__main__":
    main()
