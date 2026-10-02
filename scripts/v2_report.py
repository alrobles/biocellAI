"""Aggregate v2 run outputs into a paired-inference report (ADR-011 R5/R6).

Reads one or more run directories produced by scripts/v2_run.py (each must
contain metrics.csv and predictions_<ARM>.csv) and computes, per arm pair
(--pairs T2:B2), the paired donor-level Spearman-rho difference with a
percentile bootstrap CI via paired_rho_difference. Runs are not pooled:
each run directory contributes one paired comparison row, tagged by seed.

Usage:
    python scripts/v2_report.py --runs DIR_S0 DIR_S1 DIR_S2 \
        --pairs T2:B2 T2:T0 T2:N2 --gate-delta 0.05 --out report_dir/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biocellai.progression import paired_rho_difference
from biocellai.revalidation import file_sha256, improvement_gate


def load_predictions(run: Path, arm: str) -> pd.DataFrame:
    f = run / f"predictions_{arm}.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing — arm {arm} did not run or "
                                "produced no pathology predictions")
    d = pd.read_csv(f).set_index("donor_id")
    return d[["observed", "predicted"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="CANDIDATE:REFERENCE arm pairs, e.g. T2:B2")
    p.add_argument("--gate-delta", type=float, default=0.05,
                   help="preregistered minimum improvement in rho")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    runs = []
    for r in args.runs:
        manifest = json.loads((r / "manifest.json").read_text())
        runs.append({"dir": r, "manifest": manifest,
                     "metrics": pd.read_csv(r / "metrics.csv")})

    rows = []
    for pair in args.pairs:
        cand, _, ref = pair.partition(":")
        if not cand or not ref:
            raise ValueError(f"--pairs entries must be CANDIDATE:REFERENCE, got {pair!r}")
        for run in runs:
            seed = run["manifest"]["seed"]
            mets = run["metrics"].set_index("arm")
            try:
                a = load_predictions(run["dir"], cand)
                b = load_predictions(run["dir"], ref)
                comp = paired_rho_difference(a, b, n_boot=args.n_boot,
                                             seed=seed)
            except (FileNotFoundError, ValueError) as e:
                comp = {"delta_rho": None, "ci_low": None, "ci_high": None,
                        "n": None, "n_boot_valid": 0, "error": str(e)}
            cand_rho = mets["path_rho"].get(cand)
            ref_rho = mets["path_rho"].get(ref)
            null_rho = mets["path_rho"].get("N1")
            null_ok = (bool(null_rho <= cand_rho)
                       if pd.notna(null_rho) and pd.notna(cand_rho)
                       else None)
            comp.update({"pair": pair, "candidate": cand, "reference": ref,
                         "run": str(run["dir"]), "seed": seed,
                         "candidate_rho": cand_rho, "reference_rho": ref_rho,
                         "null_rho": null_rho, "null_ok": null_ok,
                         "gate": improvement_gate(
                             cand_rho, ref_rho, comp["ci_low"],
                             integrity_ok=comp["n_boot_valid"] > 0
                             if comp["n_boot_valid"] is not None else False,
                             null_ok=null_ok, threshold=args.gate_delta)})
            rows.append(comp)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    df.to_csv(args.out / "paired_comparisons.csv", index=False)
    report = {
        "runs": [{"dir": str(r["dir"]),
                  "manifest_sha256": file_sha256(r["dir"] / "manifest.json")}
                 for r in runs],
        "pairs": args.pairs, "gate_delta": args.gate_delta,
        "n_boot": args.n_boot,
        "comparisons": rows,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    cols = ["pair", "seed", "candidate_rho", "reference_rho", "delta_rho",
            "ci_low", "ci_high", "null_rho", "gate", "error"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
