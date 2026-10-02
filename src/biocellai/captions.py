"""Caption generation: deterministic templates from expression + markers.

Two variants by design (ADR-003):
  - include_label=True  → caption names the cell type (upper bound / leakage probe)
  - include_label=False → markers only (real grounding signal)

Marker tables are computed on TRAIN cells only — never on held-out donors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def marker_table(adata_train, groupby: str = "cell_type", n_markers: int = 5) -> dict[str, list[str]]:
    """Top marker genes per cell type, computed on train cells only.

    Uses a simple mean-expression-rank method (no scanpy dependency so this
    stays unit-testable): for each type, rank genes by mean expression in
    type minus mean outside type.
    """
    X = adata_train.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    genes = np.asarray(adata_train.var_names)
    labels = np.asarray(adata_train.obs[groupby])

    markers: dict[str, list[str]] = {}
    for ct in np.unique(labels):
        in_mask = labels == ct
        if in_mask.sum() < 2:
            markers[ct] = []
            continue
        diff = X[in_mask].mean(axis=0) - X[~in_mask].mean(axis=0)
        top = np.argsort(diff)[::-1][:n_markers]
        markers[ct] = [str(genes[i]) for i in top]
    return markers


def cell_caption(
    top_genes: list[str],
    cell_type: str | None,
    tissue: str = "blood",
    include_label: bool = True,
) -> str:
    genes_txt = ", ".join(top_genes) if top_genes else "none detected"
    if include_label and cell_type is not None:
        return f"A {cell_type} cell from human {tissue}. Highly expressed genes: {genes_txt}."
    return f"A cell from human {tissue}. Highly expressed genes: {genes_txt}."


def top_expressed_genes(X_row, var_names, k: int = 8) -> list[str]:
    x = np.asarray(X_row).ravel()
    if x.size == 0:
        return []
    k = min(k, x.size)
    idx = np.argpartition(x, -k)[-k:]
    idx = idx[np.argsort(x[idx])[::-1]]
    return [str(var_names[i]) for i in idx if x[i] > 0]


def build_cell_captions(
    adata,
    include_label: bool,
    tissue: str = "blood",
    k_genes: int = 8,
    label_key: str = "cell_type",
    mode: str = "own_genes",
    markers: dict[str, list[str]] | None = None,
    type_descriptions: dict[str, str] | None = None,
) -> list[str]:
    """One caption per cell.

    mode="own_genes"    → top expressed HVGs of the cell itself (M1).
    mode="type_markers" → the cell type's marker list computed on TRAIN cells
                          (cleaner grounding source; no label text unless
                          include_label=True). Requires `markers`.
    mode="type_llm"     → LLM-written functional description per cell type
                          (RQ3/M4: knowledge grounding). Requires
                          `type_descriptions` mapping cell_type -> prose.
                          The description does NOT repeat the label verbatim;
                          biological identity is recoverable only through
                          described function/markers.
    """
    if mode == "type_llm":
        if type_descriptions is None:
            raise ValueError("mode='type_llm' requires type_descriptions")
        labels_llm = adata.obs[label_key].astype(str).to_numpy()
        return [
            f"A cell from human {tissue}. {type_descriptions.get(labels_llm[i], 'No description.')}"
            for i in range(adata.n_obs)
        ]

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    names = np.asarray(adata.var_names)
    labels = adata.obs[label_key].astype(str).to_numpy()

    if mode == "type_markers":
        if markers is None:
            raise ValueError("mode='type_markers' requires a marker table")
        return [
            cell_caption(
                markers.get(labels[i], []),
                cell_type=labels[i] if include_label else None,
                tissue=tissue,
                include_label=include_label,
            )
            for i in range(adata.n_obs)
        ]

    return [
        cell_caption(
            top_expressed_genes(X[i], names, k=k_genes),
            cell_type=labels[i] if include_label else None,
            tissue=tissue,
            include_label=include_label,
        )
        for i in range(adata.n_obs)
    ]


def build_class_captions(
    markers: dict[str, list[str]],
    tissue: str = "blood",
) -> dict[str, str]:
    """One canonical caption per cell type for zero-shot classification."""
    return {
        ct: cell_caption(m, cell_type=ct, tissue=tissue, include_label=True)
        for ct, m in markers.items()
    }
