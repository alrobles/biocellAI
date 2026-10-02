import numpy as np

from biocellai.captions import (
    build_cell_captions,
    build_class_captions,
    cell_caption,
    marker_table,
    top_expressed_genes,
)
from conftest import make_adata


def test_marker_table_recovers_blocks():
    adata = make_adata()
    markers = marker_table(adata, n_markers=3)
    assert set(markers["T"]).issubset({f"GENE{i}" for i in range(10)})
    assert set(markers["B"]).issubset({f"GENE{i}" for i in range(10, 20)})
    assert set(markers["Mono"]).issubset({f"GENE{i}" for i in range(20, 30)})


def test_caption_label_leakage_toggle():
    with_label = cell_caption(["CD3D"], "T cell", include_label=True)
    no_label = cell_caption(["CD3D"], "T cell", include_label=False)
    assert "T cell" in with_label
    assert "T cell" not in no_label
    assert "CD3D" in no_label


def test_top_expressed_genes():
    x = np.zeros(10)
    x[[3, 7]] = [5.0, 2.0]
    names = np.array([f"g{i}" for i in range(10)])
    top = top_expressed_genes(x, names, k=3)
    assert top[0] == "g3" and "g7" in top


def test_build_cell_captions_count_and_content():
    adata = make_adata()
    caps = build_cell_captions(adata, include_label=False, k_genes=5)
    assert len(caps) == adata.n_obs
    assert all("Highly expressed genes:" in c for c in caps)


def test_class_captions_cover_types():
    adata = make_adata()
    markers = marker_table(adata, n_markers=2)
    caps = build_class_captions(markers)
    assert set(caps) == {"T", "B", "Mono"}
    assert "A T cell" in caps["T"]
