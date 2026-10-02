"""Aggregate v2 arm-matrix metrics across cohorts/seeds into tables.

Walks experiments/v2_revalidation/<cohort>/run_s<seed>_<tag>/metrics.csv,
emits a long-form CSV plus per-metric cohort x arm summary tables
(mean +- sd across seeds) for the paper/results reporting (R6-INFERENCE).

    python scripts/v2_aggregate.py --base experiments/v2_revalidation \
        --tag full --out experiments/v2_revalidation/aggregate
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RUN_RE = re.compile(r"run_s(\d+)_(.+)")


def collect(base: Path, tag: str) -> pd.DataFrame:
    rows = []
    for csv in sorted(base.glob("*/run_s*_*/metrics.csv")):
        m = RUN_RE.fullmatch(csv.parent.name)
        if not m or m.group(2) != tag:
            continue
        df = pd.read_csv(csv)
        df["cohort"] = csv.parent.parent.name
        df["seed"] = int(m.group(1))
        df["run_dir"] = str(csv.parent)
        rows.append(df)
    if not rows:
        raise SystemExit(f"no metrics.csv under {base} matching tag={tag!r}")
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("experiments/v2_revalidation"))
    p.add_argument("--tag", default="full")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    df = collect(args.base, args.tag)
    args.out.mkdir(parents=True, exist_ok=True)
    long_csv = args.out / "metrics_long.csv"
    df.to_csv(long_csv, index=False)

    metrics = [c for c in df.columns
               if c not in ("cohort", "seed", "run_dir", "arm",
                            "n_train_donors", "n_test_donors", "identity_eval")]
    arm_order = ["B0", "B1", "B2", "T0", "T1", "T2", "T3", "T4", "N1", "N2"]
    for metric in metrics:
        if df[metric].isna().all():
            continue
        g = (df.pivot_table(index=["cohort", "arm"], columns="seed",
                            values=metric, aggfunc="first")
               .reindex(columns=sorted(df.seed.unique())))
        g["mean"] = g.mean(axis=1)
        g["sd"] = g.std(axis=1)
        order = {a: i for i, a in enumerate(arm_order)}
        g = (g.reset_index()
               .sort_values(["cohort", "arm"],
                            key=lambda s: s.map(order).fillna(99)
                            if s.name == "arm" else s))
        g.to_csv(args.out / f"summary_{metric}.csv", index=False)

    manifest = {
        "step": "v2_aggregate", "tag": args.tag,
        "cohorts": sorted(df.cohort.unique()),
        "seeds": sorted(int(s) for s in df.seed.unique()),
        "arms": sorted(df.arm.unique()),
        "n_rows": len(df),
    }
    (args.out / "aggregate.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {long_csv} ({len(df)} rows, {df.cohort.nunique()} cohorts) "
          f"+ {len(metrics)} metric summaries")


if __name__ == "__main__":
    main()
