"""R5-MARKERS: non-circular marker recovery tests."""
import json

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from biocellai.marker_eval import (caption_genes, evaluate_fold,
                                   precision_at_k)


def test_precision_at_k_excludes_caption_genes():
    ranked = ["CAP1", "CAP2", "REF1", "REF2", "G1", "G2", "G3", "G4",
              "G5", "G6", "G7", "G8", "G9", "G10"]
    ref = {"REF1", "REF2"}
    out = precision_at_k(ranked, ref, exclude={"CAP1", "CAP2"}, k=10)
    assert out["candidates"][0] == "REF1"
    assert len(out["candidates"]) == 10
    assert out["precision"] == 0.2
    # without exclusion the same genes would score identically
    naive = precision_at_k(ranked, ref, k=10)
    assert naive["candidates"][0] == "CAP1"
    assert naive["precision"] == 0.2


def test_caption_genes_parses_marker_template(tmp_path):
    caps = {"Astro": "Astro: a cell type defined by elevated expression "
                     "of GFAP, AQP4, SLC1A3.",
            "Mic": "Mic: a cell type."}
    p = tmp_path / "caps.json"
    p.write_text(json.dumps(caps))
    out = caption_genes(p)
    assert out["Astro"] == {"GFAP", "AQP4", "SLC1A3"}
    assert out["Mic"] == set()


def _toy_fold(tmp_path, seed=0):
    """Two train donors + one test donor; Astro up-regulates MARKER genes."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 30, 25
    X = rng.poisson(1.0, (n_cells, n_genes)).astype(np.float32)
    ct = np.array(["Astro"] * 15 + ["Mic"] * 15)
    # Astro-elevated genes: first 12
    X[ct == "Astro", :12] += 5
    obs = pd.DataFrame({"cell_type": ct,
                        "is_test": [False] * 20 + [True] * 10})
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    ad = anndata.AnnData(X=X, obs=obs, var=var)
    p = tmp_path / f"fold_s{seed}.h5ad"
    ad.write(p)
    return p


def test_evaluate_fold_recovers_reference(tmp_path):
    fp = _toy_fold(tmp_path)
    # caption injects the top marker genes
    caps = {"Astro": "Astro: ... elevated expression of G0, G1, G2, G3, "
                     "G4, G5, G6, G7.",
            "Mic": "Mic: a cell type."}
    cp = tmp_path / "caps.json"
    cp.write_text(json.dumps(caps))
    ref = {"Astro": [f"G{i}" for i in range(12)], "Mic": ["ZZZ"]}
    res = evaluate_fold(fp, cp, ref, k=10)
    astro = next(r for r in res["classes"] if r["cell_type"] == "Astro")
    # non-circular candidates are the next Astro markers (G8..) — still
    # in the 12-gene reference -> high recovery despite exclusions
    assert astro["precision_nc"] >= 0.4
    assert set(astro["candidates_nc"]).isdisjoint(
        {"G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"})
    # Mic has a reference but no real markers -> low
    mic = next(r for r in res["classes"] if r["cell_type"] == "Mic")
    assert mic["precision_nc"] == 0.0
