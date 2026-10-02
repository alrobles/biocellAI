"""Fixture tests for the scFM embedding stack (R4-GENEFORMER).

The Geneformer tokenizer is checked against a literal port of the
upstream algorithm (geneformer.tokenizer): symbol->ensembl mapping,
duplicate-ensembl collapse, libsize*target_sum/median ranking, nonzero
filter, V2 <cls>/<eos> wrap AFTER truncating raw seq to input_size-2,
V1 no specials / 2048 cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from biocellai.scfm import gf_tokenize, load_fold_cells


def _dicts(n_genes=40):
    ens = {f"ENSG{i:05d}": 1000 + i for i in range(n_genes)}
    # distinct medians -> strict value order, no float-tie ambiguity
    med = {e: 1.0 + i * 0.013 for i, e in enumerate(ens)}
    nid = {f"G{i}": f"ENSG{i:05d}" for i in range(n_genes)}
    nid["DUP"] = "ENSG00003"  # duplicate symbol -> same ensembl
    ensmap = {}  # ensmap is the ens/alias dict; fixture is symbol-keyed
    tok = dict(ens)
    tok["<cls>"], tok["<eos>"], tok["<pad>"] = 1, 2, 0
    return tok, med, ensmap, nid


def _reference_tokenize(counts, symbols, tok, med, name2ens, special,
                        max_len, target_sum=10_000.0):
    """Literal port of geneformer.tokenizer cell loop.

    Mirrors upstream exactly: raw-key membership decides the 1:1-vs-
    collapse branch (sum_ensembl_ids); collapse emits unique-mapped rows
    in var order then summed groups sorted by ens id; .upper() lookup;
    nonzero normalized values sorted via np.argsort(-v); raw seq
    truncated to input_size-2 before CLS/EOS wrap.
    """
    from collections import Counter

    in_map = [s in name2ens for s in symbols]
    uniq_in = {s for s, im in zip(symbols, in_map) if im}
    uniq_out = {name2ens[s] for s in uniq_in}
    collapse = len(uniq_in) != len(uniq_out)
    mapped = [name2ens.get(s.upper()) for s in symbols]
    if collapse:
        cnt = Counter(e for e in mapped if e is not None)
        unique_rows = [j for j, e in enumerate(mapped)
                       if e is not None and cnt[e] == 1
                       and e in tok and e in med]
        dup_ens = sorted(e for e in cnt
                         if cnt[e] > 1 and e in tok and e in med)
        gene_cols = [[j] for j in unique_rows] + [
            [j for j, e in enumerate(mapped) if e == de] for de in dup_ens]
        ens_ids = [mapped[j] for j in unique_rows] + dup_ens
    else:
        gene_cols = [[j] for j, e in enumerate(mapped)
                     if e is not None and e in tok and e in med]
        ens_ids = [mapped[j] for j, e in enumerate(mapped)
                   if e is not None and e in tok and e in med]
    inner = max_len - 2 if special else max_len
    seqs = []
    for row in np.asarray(counts, dtype=np.float64):
        libsize = row.sum() or 1.0
        vals = np.array([row[c].sum() / libsize * target_sum / med[e]
                         for e, c in zip(ens_ids, gene_cols)])
        nz = vals > 0
        order = np.argsort(-vals[nz])
        seq = [tok[ens_ids[i]] for i in np.flatnonzero(nz)[order]][:inner]
        if special:
            seq = [tok["<cls>"], *seq, tok["<eos>"]]
        seqs.append(np.asarray(seq))
    return seqs


@pytest.fixture
def tiny_adata():
    rng = np.random.default_rng(0)
    symbols = [f"G{i}" for i in range(30)] + ["DUP", "UNMAPPED"]
    X = sparse.csr_matrix(rng.poisson(0.8, (12, 32)).astype(np.float32))
    obs = pd.DataFrame({"donor_id": ["d0", "d1"] * 6,
                        "cell_type": ["t", "b"] * 6},
                       index=[f"c{i}" for i in range(12)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=symbols))


@pytest.mark.parametrize("special,max_len", [(False, 2048), (True, 4096)])
def test_gf_tokenize_matches_reference(tiny_adata, special, max_len):
    tok, med, ensmap, nid = _dicts()
    got = gf_tokenize(tiny_adata, tok, med, ensmap, nid, special, max_len,
                      dict_choice="nid")
    want = _reference_tokenize(tiny_adata.X.toarray(),
                               tiny_adata.var_names.tolist(),
                               tok, med, nid, special, max_len)
    assert len(got) == tiny_adata.n_obs
    for i, (g, w) in enumerate(zip(got, want)):
        assert np.array_equal(g, w), f"cell {i}: {g[:8]} != {w[:8]}"


def test_gf_tokenize_collapse_and_truncate(tiny_adata):
    tok, med, ensmap, nid = _dicts()
    # force dense expression so sequences overflow max_len
    adata = tiny_adata.copy()
    adata.X = sparse.csr_matrix(
        np.ones_like(adata.X.toarray(), dtype=np.float32) * 5)
    got = gf_tokenize(adata, tok, med, ensmap, nid, special=True,
                      max_len=10, dict_choice="nid")
    want = _reference_tokenize(adata.X.toarray(), adata.var_names.tolist(),
                               tok, med, nid, True, 10)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)
        assert len(g) == 10 and g[0] == tok["<cls>"] and g[-1] == tok["<eos>"]
    # duplicate symbol "DUP" must collapse into ENSG00003 (single token)
    got_nodup = gf_tokenize(tiny_adata, tok, med, ensmap, nid, False, 2048,
                            dict_choice="nid")
    for seq in got_nodup:
        assert list(seq).count(tok["ENSG00003"]) <= 1


def test_gf_tokenize_ties_stable(tiny_adata):
    """Fully tied values: token multiset must match; order may legitimately
    differ by float-ulp (upstream has the same tie ambiguity)."""
    from collections import Counter
    tok, med, ensmap, nid = _dicts()
    a = tiny_adata[:2].copy()
    a.X = sparse.csr_matrix(np.full((2, a.n_vars), 2.0, dtype=np.float32))
    got = gf_tokenize(a, tok, med, ensmap, nid, False, 2048,
                      dict_choice="nid")
    want = _reference_tokenize(a.X.toarray(), a.var_names.tolist(),
                               tok, med, nid, False, 2048)
    for g, w in zip(got, want):
        assert Counter(g.tolist()) == Counter(w.tolist())


def test_load_fold_cells_alignment(tmp_path, tiny_adata):
    ext = tiny_adata.copy()
    ext.write_h5ad(tmp_path / "ext.h5ad")
    fold = tiny_adata[[4, 0, 9]].copy()
    fold.obs["extra"] = [1, 2, 3]
    fold.write_h5ad(tmp_path / "fold.h5ad")
    out = load_fold_cells(tmp_path / "ext.h5ad", tmp_path / "fold.h5ad",
                          "native")
    assert out.obs_names.tolist() == ["c4", "c0", "c9"]
    assert "extra" in out.obs.columns
    hvg = load_fold_cells(tmp_path / "ext.h5ad", tmp_path / "fold.h5ad",
                          "hvg")
    assert hvg.n_vars <= fold.n_vars
