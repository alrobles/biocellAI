#!/usr/bin/env python3
"""M5: consolidate all experiment metrics into one comparison table.

Reads metrics.csv from each experiments/* directory and emits a combined
CSV + markdown table, tagging each run with its experiment/scale and
flagging invalid (pre-gene-symbol-fix) or leakage-probe rows.

Usage:
    python scripts/consolidate.py \
        --dirs experiments/m1_grounding_benchmark \
               experiments/m3_ablation_85k \
               experiments/m4_llm_honest_85k \
               experiments/m4_llm_labeled_85k \
        --out experiments/m5_consolidated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_run(d: Path) -> pd.DataFrame | None:
    f = d / "metrics.csv"
    if not f.exists() or (d / "SUPERSEDED.md").exists():
        return None
    df = pd.read_csv(f)
    df["experiment"] = d.name
    for col in ("caption_mode", "text_model", "label_mode", "arm"):
        if col not in df.columns:
            df[col] = "-"
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    frames = [df for d in args.dirs if (df := load_run(d)) is not None]
    skipped = [d.name for d in args.dirs if load_run(d) is None]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["leakage"] = (
        all_df.label_mode.astype(str).eq("True")
        | all_df.caption_mode.astype(str).str.contains("labeled", na=False)
        | (all_df.experiment.str.contains("labeled", na=False)
           & (all_df.arm != "cell_only_supervised"))
    )

    keys = ["experiment", "arm", "caption_mode", "text_model", "leakage"]
    summary = (
        all_df.groupby(keys, dropna=False)[["macro_f1", "balanced_acc"]]
        .agg(["mean", "std"]).round(4)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out / "all_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv")

    lines = ["# M5 — consolidated results", "",
             f"Runs: {[d.name for d in args.dirs]}",
             f"Skipped (superseded/missing): {skipped}", "",
             "`leakage=True` rows are probes/upper bounds, not honest grounding.", "",
             "```", summary.to_string(), "```", ""]
    (args.out / "README.md").write_text("\n".join(lines))
    print(summary.to_string())
    print(f"\nskipped: {skipped}")


if __name__ == "__main__":
    main()
