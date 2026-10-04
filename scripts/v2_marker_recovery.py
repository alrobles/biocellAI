#!/usr/bin/env python3
"""R5-MARKERS: marker recovery vs external references, without circularity.

For each fold: rerank train-only Wilcoxon markers, drop the genes that
were injected into that fold's marker captions, and score Precision@10
against an external canonical reference (PanglaoDB manifests). Gate:
macro P@10 >= 0.6.

Usage:
    python scripts/v2_marker_recovery.py \
        --fold experiments/v2_revalidation/seaad_mtg/folds/fold_s0.h5ad \
        --captions experiments/v2_revalidation/seaad_mtg/folds/captions_s0_markers.json \
        --reference data/manifests/panglao_brain_markers.json \
        --cohort seaad_mtg --seed 0 --out experiments/v2_marker_recovery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from biocellai import marker_eval  # noqa: E402
from biocellai.revalidation import file_sha256  # noqa: E402

GATE = 0.6


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=Path, action="append", required=True)
    p.add_argument("--captions", type=Path, action="append", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    if len(args.fold) != len(args.captions):
        p.error("--fold and --captions must have the same length")

    ref = json.loads(args.reference.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    rows, details, seeds = [], {}, []
    for fold_path, cap_path in zip(args.fold, args.captions):
        seed = int(fold_path.stem.rsplit("_s", 1)[1])
        seeds.append(seed)
        res = marker_eval.evaluate_fold(fold_path, cap_path, ref, k=args.k)
        details[f"s{seed}"] = res
        m = res["macro"]
        rows.append({"cohort": args.cohort, "seed": seed,
                     "n_classes": m["n_classes"],
                     "n_evaluable": m["n_evaluable"],
                     "precision_nc": m["precision_nc"],
                     "precision_naive": m["precision_naive"]})

    df = pd.DataFrame(rows).sort_values("seed")
    cohort_mean = df.precision_nc.mean()
    gate = "PASS" if cohort_mean >= GATE else "FAIL"

    df.to_csv(args.out / f"{args.cohort}_per_seed.csv", index=False)
    (args.out / f"{args.cohort}_classes.json").write_text(
        json.dumps(details, indent=1))

    manifest = {
        "task": "R5-MARKERS",
        "cohort": args.cohort,
        "seeds": seeds,
        "k": args.k,
        "gate_threshold": GATE,
        "gate": gate,
        "macro_precision_nc": cohort_mean,
        "n_evaluable_total": int(df.n_evaluable.sum()),
        "design": ("Candidates = train-only Wilcoxon marker ranking minus "
                   "genes injected into the fold's marker captions; "
                   "reference = external PanglaoDB canonical sets"),
        "folds": [file_sha256(f) for f in args.fold],
        "captions": [file_sha256(c) for c in args.captions],
        "reference_file": str(args.reference),
        "reference_sha256": file_sha256(args.reference),
    }
    (args.out / f"{args.cohort}_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"{args.cohort}: macro P@{args.k} (non-circular) = "
          f"{cohort_mean:.3f} over {len(df)} seeds -> {gate}")


if __name__ == "__main__":
    main()
