"""R6-INFERENCE: pooled paired comparisons, permutation nulls, gates.

Aggregates per-donor predictions across seeds with DONOR-CLUSTERED
bootstrap (a donor repeated across seed folds is one resampling unit),
loads the shared 100-map permutation nulls (v2_perm_null.py), applies
BH-FDR over secondary comparisons, and emits paired_comparisons.csv +
gates.json.

Usage:
    python scripts/v2_inference.py --root experiments/v2_revalidation \
        --cohorts seaad_mtg:cps_global rosmap:path_level \
        --primary-cohort seaad_mtg --n-boot 2000 \
        --out experiments/v2_revalidation/inference
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from biocellai import inference as _inf  # noqa: E402
from biocellai.revalidation import file_sha256  # noqa: E402

ARMS = ["B0", "B1", "B2", "T0", "T1", "T2", "T3", "N1", "N2"]
SEEDS = (0, 1, 2)
# (candidate, reference): every candidate vs B2 and T1; baselines; sanity
PAIRS = ([(c, r) for c in ("T0", "T2", "T3", "N1", "N2") for r in ("B2", "T1")]
         + [("B2", "B0"), ("B2", "B1"), ("T2", "T0"), ("T3", "T0"),
            ("N2", "T2"), ("T2", "N2"), ("T3", "N2")])
PRIMARY_PAIRS = {("T2", "B2"), ("T3", "B2"), ("T2", "T1"), ("T3", "T1")}


def _arm_rho(pred: pd.DataFrame) -> float:
    from scipy.stats import spearmanr
    if len(pred) < 3:
        return np.nan
    return float(spearmanr(pred.observed, pred.predicted).statistic)


def _check_run_integrity(run_dir: Path, root: Path, cohort: str) -> list[str]:
    """Fold sha256 recorded vs actual file; split disjointness is
    enforced upstream by prepare_donor_fold (is_test mask)."""
    errs = []
    man_p = run_dir / "manifest.json"
    if not man_p.exists():
        return [f"{run_dir}: no manifest"]
    man = json.loads(man_p.read_text())
    fold_name = Path(man["fold"]["path"]).name
    candidates = [root / cohort / "folds" / fold_name,
                  root / cohort / fold_name]
    fold = next((f for f in candidates if f.exists()), None)
    if fold is None:
        errs.append(f"{run_dir}: fold {fold_name} not found under {cohort}")
    elif file_sha256(fold) != man["fold"]["sha256"]:
        errs.append(f"{run_dir}: fold sha256 mismatch")
    return errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cohorts", nargs="+", required=True,
                   metavar="NAME:PATH_COL[:SEED,SEED,...]")
    p.add_argument("--primary-cohort", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=12345)
    p.add_argument("--transfer-metrics", type=Path, default=None,
                   help="transfer metrics_long.csv (strict rhos feed gate)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root

    cohorts = {}
    for spec in args.cohorts:
        parts = spec.split(":")
        name = parts[0]
        col = parts[1] if len(parts) > 1 else None
        seeds = ([int(x) for x in parts[2].split(",")]
                 if len(parts) > 2 else list(SEEDS))
        cohorts[name] = {"path_col": col or None, "seeds": seeds}

    # ---------------- collect predictions + integrity --------------------
    integrity_errs = []
    matrix_missing = []
    preds = {}      # preds[cohort][arm][seed] -> df
    arm_rhos = {}   # arm_rhos[cohort][arm] -> mean rho over seeds
    nulls = {}      # nulls[cohort] -> df
    for cohort, cinfo in cohorts.items():
        preds[cohort], arm_rhos[cohort] = {}, {}
        null_parts = []
        for seed in cinfo["seeds"]:
            run = root / cohort / f"run_s{seed}_full"
            integrity_errs += _check_run_integrity(run, root, cohort)
            metrics = (pd.read_csv(run / "metrics.csv")
                       if (run / "metrics.csv").exists()
                       else pd.DataFrame())
            for arm in ARMS:
                pf = run / f"predictions_{arm}.csv"
                has_pred = pf.exists()
                has_metric = (not metrics.empty
                              and (metrics.arm == arm).any())
                if not (has_pred and has_metric):
                    matrix_missing.append(f"{cohort}/s{seed}/{arm}")
                    continue
                preds[cohort].setdefault(arm, {})[seed] = \
                    _inf.load_predictions(run, arm)
            pn = root / cohort / f"perm_null_s{seed}"
            if pn.is_dir():
                nd = _inf.load_perm_nulls(pn)
                if not nd.empty:
                    nd["seed"] = seed
                    null_parts.append(nd)
        for arm, per_seed in preds[cohort].items():
            arm_rhos[cohort][arm] = float(np.nanmean(
                [_arm_rho(d) for d in per_seed.values()]))
        nulls[cohort] = (pd.concat(null_parts, ignore_index=True)
                         if null_parts else pd.DataFrame(
                             columns=["arm", "perm", "rho", "n_test",
                                      "seed"]))

    # ---------------- pooled paired comparisons --------------------------
    rows = []
    for cohort in cohorts:
        for cand, ref in PAIRS:
            if cand not in preds[cohort] or ref not in preds[cohort]:
                continue
            res = _inf.clustered_delta_rho(
                preds[cohort][cand], preds[cohort][ref],
                n_boot=args.n_boot,
                seed=args.boot_seed
                + zlib.crc32(f"{cohort}:{cand}:{ref}".encode()) % 99991)
            role = ("primary" if cohort == args.primary_cohort
                    and (cand, ref) in PRIMARY_PAIRS else "secondary")
            rows.append({"cohort": cohort, "candidate": cand,
                         "reference": ref, "role": role,
                         "delta_rho": res["delta_rho"],
                         "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                         "p_emp": res["p_emp"],
                         "n_boot_valid": res["n_boot_valid"],
                         "n_donors": res["n_donors"],
                         "candidate_rho": arm_rhos[cohort].get(cand),
                         "reference_rho": arm_rhos[cohort].get(ref),
                         "per_seed_delta": json.dumps(res["per_seed"])})
    comp_df = pd.DataFrame(rows)
    # BH-FDR across secondary comparisons (per cohort)
    comp_df["q_fdr"] = np.nan
    for cohort in cohorts:
        sec = ((comp_df.cohort == cohort) & (comp_df.role == "secondary"))
        comp_df.loc[sec, "q_fdr"] = _inf.benjamini_hochberg(
            comp_df.loc[sec, "p_emp"])
    comp_df["comparison"] = comp_df.candidate + "-" + comp_df.reference

    # ---------------- gates ----------------------------------------------
    prim = args.primary_cohort
    prim_comp = (comp_df[comp_df.cohort == prim]
                 .set_index("comparison"))
    transfer_rhos = {}
    if args.transfer_metrics and Path(args.transfer_metrics).exists():
        tm = pd.read_csv(args.transfer_metrics)
        strict = tm[(tm["mode"] == "strict") & (tm["arm"] == "B2")
                    & (tm["eval_subset"] == "all_donors")]
        for direction, g in strict.groupby("direction"):
            transfer_rhos[direction] = g["rho"].tolist()

    composition = None
    resid = root / prim / "state_analysis" / "residuals.csv"
    if resid.exists():
        rd = pd.read_csv(resid)
        rd = rd[rd.state == "state_scfm"]
        if not rd.empty:
            composition = {
                "delta_rho": float(rd.delta_rho.mean()),
                "ci_low": float(rd.ci_lo.min()),
                "per_seed": rd[["seed", "delta_rho", "ci_lo"]].to_dict(
                    "records")}

    gates = _inf.evaluate_gates(
        integrity_ok=not integrity_errs,
        comparisons=prim_comp,
        arm_rhos=arm_rhos.get(prim, {}),
        nulls=nulls.get(prim, pd.DataFrame()),
        transfer_rhos=transfer_rhos,
        composition=composition,
        matrix_complete=not matrix_missing)
    gates["_integrity_errors"] = integrity_errs
    gates["_matrix_missing"] = matrix_missing
    gates["_null_counts"] = {c: {a: int(((nulls[c].arm == a)).sum())
                                 for a in ARMS} for c in cohorts
                             if not nulls[c].empty}

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(out / "paired_comparisons.csv", index=False)
    (out / "gates.json").write_text(json.dumps(gates, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "protocol": "v2_inference", "root": str(root.resolve()),
        "cohorts": cohorts, "primary_cohort": prim,
        "n_boot": args.n_boot, "boot_seed": args.boot_seed,
        "bootstrap": ("donor-clustered: donor identity is the resampling "
                      "unit across seeds; statistic = mean per-seed "
                      "paired delta-rho on shared test donors"),
        "multiplicity": ("BH-FDR across secondary comparisons per cohort; "
                         "primary comparisons unadjusted"),
        "perm_null_design": ("100 shared train-donor->label maps per "
                             "cohort x seed; B2 retrains on permuted "
                             "labels, other arms permute the readout"),
    }, indent=2))
    print(comp_df.to_string(index=False))
    print("\n" + json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
