"""Build fold-specific caption banks for the v2 text arms (ADR-011 R1-CAPTIONS).

Captions must never incorporate information from test donors. Two sources:

  markers   differential markers computed on TRAIN cells only, rendered as
            template prose -> caption (arm T2)
  copy      adopt an existing corpus file (e.g. PubMed retrieval built from
            external literature). Coverage is checked against the fold's
            classes; use --missing-policy name to fill gaps with class-name
            captions (recorded in provenance) or fail.
  name      class-name-only captions (degraded-text arm / sanity check)

Writes <out>/captions_s{seed}_{mode}.json plus .prov.json per fold file
supplied via --fold (repeatable).
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np

from biocellai.revalidation import file_sha256


def marker_captions(fold, label_col: str, top_n: int) -> dict:
    import scanpy as sc

    tr = fold[~fold.obs["is_test"].to_numpy()].copy()
    sc.tl.rank_genes_groups(tr, label_col, method="wilcoxon")
    caps = {}
    for cls in tr.uns["rank_genes_groups"]["names"].dtype.names:
        genes = [g for g in
                 tr.uns["rank_genes_groups"]["names"][cls][:top_n]]
        genes = [g for g in genes if isinstance(g, str)]
        caps[cls] = (f"{cls}: a cell type defined by elevated expression of "
                     f"{', '.join(genes)}.")
    return caps


def name_captions(classes: list[str]) -> dict:
    return {c: f"{c}: a cell type." for c in classes}


def copy_captions(path: Path, classes: list[str], missing_policy: str) -> tuple[dict, list[str]]:
    src = json.loads(Path(path).read_text())
    caps = {c: src[c] for c in classes if c in src}
    missing = [c for c in classes if c not in src]
    if missing and missing_policy == "fail":
        raise ValueError(f"{path}: no caption for classes {missing}")
    caps.update({c: f"{c}: a cell type." for c in missing})
    return caps, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=Path, action="append", required=True,
                   help="prepared fold h5ad; repeatable")
    p.add_argument("--mode", choices=["markers", "copy", "name"],
                   required=True)
    p.add_argument("--source", type=Path, default=None,
                   help="caption corpus JSON (required for mode=copy)")
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--missing-policy", choices=["fail", "name"], default="fail")
    p.add_argument("--tag", default=None,
                   help="output filename tag (defaults to mode)")
    args = p.parse_args()

    for fold_path in args.fold:
        seed = fold_path.stem.rsplit("_s", 1)[-1]
        fold = ad.read_h5ad(fold_path)
        classes = sorted(fold.obs[args.label_col].astype(str).unique())
        prov = {"step": "v2_captions", "mode": args.mode,
                "fold": str(fold_path.resolve()),
                "fold_sha256": file_sha256(fold_path),
                "label_col": args.label_col, "n_classes": len(classes),
                "classes": classes,
                "environment": {n: version(n) for n in ("anndata", "scanpy")}}

        if args.mode == "markers":
            caps = marker_captions(fold, args.label_col, args.top_n)
            prov.update({"fit_scope": "train_cells_only", "top_n": args.top_n,
                         "method": "wilcoxon_rank_genes_groups"})
        elif args.mode == "name":
            caps = name_captions(classes)
            prov.update({"fit_scope": "none"})
        else:
            if args.source is None:
                raise ValueError("--source required for mode=copy")
            caps, missing = copy_captions(args.source, classes,
                                          args.missing_policy)
            prov.update({"source_file": str(args.source.resolve()),
                         "source_sha256": file_sha256(args.source),
                         "missing_policy": args.missing_policy,
                         "missing_classes": missing})

        tag = args.tag or args.mode
        out = fold_path.parent / f"captions_s{seed}_{tag}.json"
        prov_path = out.with_suffix(".prov.json")
        if out.exists() or prov_path.exists():
            raise FileExistsError(f"refusing to overwrite {out}")
        out.write_text(json.dumps(caps, indent=1, ensure_ascii=False))
        prov["captions_sha256"] = file_sha256(out)
        prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False))
        print(f"wrote {out} ({len(caps)} classes)")


if __name__ == "__main__":
    main()
