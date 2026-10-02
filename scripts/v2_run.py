"""Run the v2 arm matrix on one prepared fold (ADR-011 R3).

Usage:
    python scripts/v2_run.py --fold experiments/v2_revalidation/run/fold_s0.h5ad \
        --out experiments/v2_revalidation/run/s0 \
        --arms B0,B1,B2,T0,T1,N1 \
        --label-col cell_type --pathology-col cps_global \
        [--captions T2=caps_manual.json] [--captions T3=caps_pubmed.json] \
        [--n2-of T2] [--text-encoder all-MiniLM-L6-v2]

Caption files are JSON {class_label: caption_text}. Text arms T2/T3/T4 differ
only in caption provenance; N2 reuses the --n2-of arm's captions with a
permuted class->caption map. The output directory is created once and is
refused if it already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from biocellai import data as _data  # noqa: E402
from biocellai import train as _train, v2run as _v2run  # noqa: E402
from biocellai.revalidation import (  # noqa: E402
    file_sha256,
    reserve_output,
)


def _load_text_targets(args, cats):
    """Caption files -> per-arm class embedding banks (+ permuted N2 bank)."""
    targets = {}
    encoder = None
    for spec in args.captions or []:
        arm, _, path = spec.partition("=")
        arm = arm.strip().upper()
        if not path:
            raise ValueError(f"--captions must be ARM=PATH, got {spec!r}")
        caps = json.loads(Path(path).read_text())
        missing = [c for c in cats if c not in caps]
        if missing:
            raise ValueError(f"{path}: captions missing for classes {missing}")
        if encoder is None:
            from biocellai.model import TextEncoder
            encoder = TextEncoder(args.text_encoder, device=args.device)
        bank = encoder.embed([caps[c] for c in cats])
        targets[arm] = {"bank": bank, "captions_file": str(path),
                        "captions_sha256": file_sha256(path)}
    if "N2" in [a.strip().upper() for a in args.arms.split(",")]:
        src = args.n2_of or next(iter(targets), None)
        if src not in targets:
            raise ValueError("N2 needs --n2-of <arm> with a loaded caption file")
        bank = targets[src]["bank"]
        perm = torch.from_numpy(
            np.random.default_rng(args.seed + 888).permutation(bank.shape[0]))
        targets["N2"] = {"bank": bank[perm],
                         "permutes": src, "seed": args.seed + 888}
    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", default=",".join(_v2run.CORE_ARMS))
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--pathology-col", default=None)
    p.add_argument("--covariates", default="")
    p.add_argument("--captions", action="append", metavar="ARM=PATH")
    p.add_argument("--n2-of", default=None)
    p.add_argument("--text-encoder", default="all-MiniLM-L6-v2")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden", default="256,128")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--proto-dim", type=int, default=384)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    fold = ad.read_h5ad(args.fold)
    if "is_test" not in fold.obs:
        raise ValueError("fold lacks obs['is_test'] — run v2_prepare.py first")

    labels = fold.obs[args.label_col].astype(str)
    cats = sorted(labels.unique())
    text_targets = _load_text_targets(args, cats)

    tc = _train.TrainConfig(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, hidden=tuple(int(h) for h in args.hidden.split(",")),
        embed_dim=args.embed_dim, dropout=args.dropout, device=args.device)
    cfg = _v2run.V2RunConfig(
        train=tc, label_col=args.label_col, pathology_col=args.pathology_col,
        covariates=tuple(c for c in args.covariates.split(",") if c),
        proto_dim=args.proto_dim, ridge_alpha=args.ridge_alpha, seed=args.seed)

    metrics_df, predictions, states = _v2run.run_fold(
        fold, arms, cfg, text_targets=text_targets)

    manifest = {
        "protocol": "v2", "arms": arms,
        "fold": {"path": str(args.fold.resolve()),
                 "sha256": file_sha256(args.fold)},
        "config": {"label_col": args.label_col,
                   "pathology_col": args.pathology_col,
                   "covariates": cfg.covariates, "proto_dim": args.proto_dim,
                   "ridge_alpha": args.ridge_alpha,
                   "train": {k: str(v) for k, v in vars(tc).items()}},
        "text_targets": {a: {k: v for k, v in t.items() if k != "bank"}
                         for a, t in text_targets.items()},
        "text_encoder": (args.text_encoder if text_targets else None),
        "environment": {name: version(name) for name in
                        ("numpy", "pandas", "torch", "scanpy", "anndata")},
        "code_sha256": {"v2run.py": file_sha256(_v2run.__file__),
                        "train.py": file_sha256(_train.__file__),
                        "data.py": file_sha256(_data.__file__),
                        "v2_run.py": file_sha256(__file__)},
        "seed": args.seed,
    }
    out = reserve_output(args.out)
    _v2run.write_run(out, metrics_df, predictions, states, manifest)
    print(metrics_df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
