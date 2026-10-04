"""R5-MARKERS: non-circular marker recovery evaluation.

The marker captions (T2 arm) are built from the top-N train-derived
Wilcoxon marker genes per cell type. Scoring the caption pipeline by
"do the recovered markers match canonical markers" while counting those
same injected genes would be circular. This module re-ranks train markers
and evaluates Precision@k on the ranking with caption genes excluded,
against an EXTERNAL reference (PanglaoDB canonical markers).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def train_marker_ranks(adata, label_col: str = "cell_type",
                       train_col: str = "is_test") -> dict[str, list[str]]:
    """Wilcoxon one-vs-rest ranking per class on TRAIN cells only.

    Same method as `scripts/v2_captions.marker_captions`, so the ranking is
    identical to the one that produced the caption genes — ranks beyond
    `top_n` come from the same call.
    """
    import scanpy as sc

    tr = adata[~adata.obs[train_col].to_numpy()].copy()
    sc.tl.rank_genes_groups(tr, label_col, method="wilcoxon")
    return {cls: [str(g) for g in tr.uns["rank_genes_groups"]["names"][cls]
                  if isinstance(g, str)]
            for cls in tr.uns["rank_genes_groups"]["names"].dtype.names}


_CAPTION_RE = re.compile(r"elevated expression of ([^.\n]+)\.?")


def caption_genes(captions_path: str | Path) -> dict[str, set[str]]:
    """Extract the marker genes named in each class caption."""
    caps = json.loads(Path(captions_path).read_text())
    out: dict[str, set[str]] = {}
    for cls, text in caps.items():
        m = _CAPTION_RE.search(str(text))
        out[cls] = ({g.strip() for g in m.group(1).split(",") if g.strip()}
                    if m else set())
    return out


def precision_at_k(ranked: list[str], reference: set[str],
                   exclude: set[str] | None = None,
                   k: int = 10) -> dict:
    """P@k of `ranked` after dropping `exclude` genes from the candidate list."""
    cand = [g for g in ranked if g not in (exclude or set())][:k]
    hits = [g for g in cand if g in reference]
    return {"precision": len(hits) / k, "candidates": cand, "hits": hits,
            "n_candidates": len(cand)}


def evaluate_fold(fold_path: str | Path, captions_path: str | Path,
                  reference: dict[str, list[str]], k: int = 10,
                  label_col: str = "cell_type") -> dict:
    """Evaluate one fold: per-class non-circular P@k + macro summary."""
    import anndata as ad

    fold = ad.read_h5ad(fold_path)
    space = set(map(str, fold.var_names))
    ranks = train_marker_ranks(fold, label_col=label_col)
    cap_genes = caption_genes(captions_path)

    rows = []
    for cls, ranked in ranks.items():
        ref = set(reference.get(cls, []))
        ref_space = ref & space
        cap = cap_genes.get(cls, set())
        noncirc = precision_at_k(ranked, ref, exclude=cap, k=k)
        naive = precision_at_k(ranked, ref, exclude=set(), k=k)
        rows.append({"cell_type": cls,
                     "n_reference": len(ref),
                     "n_ref_in_space": len(ref_space),
                     "ceiling_nc": min(k, len(ref_space)) / k,
                     "n_caption_genes": len(cap),
                     "precision_nc": noncirc["precision"],
                     "hits_nc": noncirc["hits"],
                     "candidates_nc": noncirc["candidates"],
                     "precision_naive": naive["precision"],
                     "evaluable": len(ref) > 0})

    ev = [r for r in rows if r["evaluable"]]
    return {
        "classes": rows,
        "macro": {
            "n_classes": len(rows),
            "n_evaluable": len(ev),
            "precision_nc": float(np.mean([r["precision_nc"]
                                           for r in ev])) if ev else None,
            "precision_naive": float(np.mean([r["precision_naive"]
                                              for r in ev])) if ev else None,
            "ceiling_nc": float(np.mean([r["ceiling_nc"]
                                         for r in ev])) if ev else None,
            "n_ref_in_space": int(sum(r["n_ref_in_space"] for r in ev)),
        },
    }
