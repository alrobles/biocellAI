"""R2-TRANSFER — strict source-fitted cross-cohort transfer + calibrated refit.

Usage:
    python scripts/v2_transfer.py \
        --source-fold  experiments/v2_revalidation/rosmap/folds/fold_s0.h5ad \
        --source-run   experiments/v2_revalidation/rosmap/run_s0_full \
        --target-raw   /path/to/v2_src/seaad_mtg.h5ad \
        --target-fold  experiments/v2_revalidation/seaad_mtg/folds/fold_s0.h5ad \
        --source-cohort rosmap --target-cohort seaad_mtg \
        --target-pathology CPS_Global \
        --out experiments/v2_revalidation/transfer/rosmap_to_seaad_mtg/s0

STRICT rows apply the source-fitted pipeline untouched to every target donor.
REFIT rows re-orient the readout on target TRAIN donors and evaluate on target
TEST donors — the calibrated analysis ADR-011 keeps separate from strict
transfer. The output directory is created once and refused if it exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from biocellai import transfer as _tr  # noqa: E402
from biocellai.revalidation import (  # noqa: E402
    file_sha256,
    reserve_output,
)


def _git_rev():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-fold", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True,
                   help="run dir containing manifest.json + ckpt_<arm>.pt")
    p.add_argument("--target-raw", type=Path, required=True,
                   help="verified-counts extract h5ad for the target cohort")
    p.add_argument("--target-fold", type=Path, required=True)
    p.add_argument("--source-cohort", required=True)
    p.add_argument("--target-cohort", required=True)
    p.add_argument("--target-pathology", required=True)
    p.add_argument("--arms", default=",".join(_tr.ALL_TRANSFER_ARMS))
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="source split/PCA seed bookkeeping (recorded)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    src_manifest = json.loads(
        (args.source_run / "manifest.json").read_text())
    src_path_col = src_manifest["config"]["pathology_col"]
    if not src_path_col:
        raise ValueError("source run has no pathology_col — nothing to transfer")

    src = ad.read_h5ad(args.source_fold)
    tgt_fold = ad.read_h5ad(args.target_fold, backed="r")
    raw = ad.read_h5ad(args.target_raw)
    tgt_X, gene_stats = _tr.target_matrix_on_source_genes(
        raw, tgt_fold.obs_names, list(src.var_names.astype(str)))
    tgt_obs = tgt_fold.obs.copy()
    if "is_test" not in tgt_obs:
        raise ValueError("target fold lacks obs['is_test']")

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    rows, infos = [], {}
    out = reserve_output(args.out)
    for arm in arms:
        if arm in _tr.ENCODER_ARMS and not (
                args.source_run / f"ckpt_{arm}.pt").exists():
            rows.append(dict(arm=arm, mode="strict", eval_subset="all_donors",
                             rho=float("nan"), r2=float("nan"),
                             mae=float("nan"), n=0,
                             note="no checkpoint"))
            continue
        r, preds, info = _tr.transfer_arm(
            arm, src, args.source_run, src_path_col, args.label_col,
            ridge_alpha=float(
                src_manifest["config"].get("ridge_alpha", 1.0)),
            seed=args.seed, device=args.device,
            tgt_X=tgt_X, tgt_obs=tgt_obs,
            tgt_path_col=args.target_pathology)
        rows.extend(r)
        infos[arm] = info
        for mode, df in preds.items():
            df.to_csv(out / f"predictions_{arm}_{mode}.csv", index=False)
        print(arm, json.dumps({rr["mode"] + "/" + rr["eval_subset"]:
                               round(rr["rho"], 4) for rr in r}), flush=True)

    manifest = {
        "step": "v2_transfer", "protocol": "v2",
        "mode_definitions": {
            "strict": "encoder, normalization, axis/predictor and sign fitted "
                      "on source cohort only; zero target fitting/labels",
            "refit": "source features, donor readout refit on target TRAIN "
                     "donors; evaluated on target TEST donors (calibrated "
                     "orientation — separate analysis)",
        },
        "source": {"cohort": args.source_cohort,
                   "fold": str(args.source_fold.resolve()),
                   "fold_sha256": file_sha256(args.source_fold),
                   "run_dir": str(args.source_run.resolve()),
                   "run_manifest_sha256": file_sha256(
                       args.source_run / "manifest.json"),
                   "pathology_col": src_path_col},
        "target": {"cohort": args.target_cohort,
                   "fold": str(args.target_fold.resolve()),
                   "fold_sha256": file_sha256(args.target_fold),
                   "raw": str(args.target_raw.resolve()),
                   "raw_sha256": file_sha256(args.target_raw),
                   "pathology_col": args.target_pathology},
        "gene_stats": gene_stats,
        "arms": arms, "label_col": args.label_col, "seed": args.seed,
        "per_arm_info": infos,
        "environment": {name: version(name) for name in
                        ("numpy", "pandas", "torch", "anndata", "scipy",
                         "scikit-learn")},
        "code_sha256": {"transfer.py": file_sha256(_tr.__file__),
                        "v2_transfer.py": file_sha256(__file__)},
        "git": _git_rev(),
        "wall_s": round(time.time() - t0, 1),
    }
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
