"""M12 — input-emb plumbing (S1) + scGPT cellset tokenisation (S2)."""
import numpy as np
import pytest
from conftest import make_adata
from scipy import sparse

from biocellai.experiment import load_input_emb
from biocellai.scgpt_enc import ScgptCellSet, pd_groupby_positions


def test_cellset_cls_prepend_and_nonzero():
    X = sparse.csr_matrix(np.array([[0, 1, 0, 2],
                                    [3, 0, 0, 0]], dtype=np.float32))
    ds = ScgptCellSet(X, gene_ids=np.array([10, 11, 12, 13]),
                      cls_id=0, pad_value=-2)
    item = ds[0]
    assert item["id"] == 0
    assert item["genes"].tolist() == [0, 11, 13]
    assert item["expressions"].tolist() == [-2.0, 1.0, 2.0]


def test_cellset_index_maps_to_global_rows():
    X = np.array([[0, 1], [3, 0], [0, 0]], dtype=np.float32)
    ds = ScgptCellSet(X, gene_ids=np.array([10, 11]), cls_id=0,
                      pad_value=-2, index=np.array([2, 1]))
    assert len(ds) == 2
    assert ds[0]["genes"].tolist() == [0]          # row 2 all-zero -> <cls> only
    assert ds[1]["genes"].tolist() == [0, 10]      # row 1
    assert ds[1]["expressions"].tolist() == [-2.0, 3.0]


def test_pd_groupby_positions():
    g = pd_groupby_positions(np.array(["a", "b", "a", "c"]))
    assert sorted(tuple(sorted(x.tolist())) for x in g) == \
        sorted([(0, 2), (1,), (3,)])


def test_load_input_emb_obs_names_realign(tmp_path):
    adata = make_adata()
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    emb = np.arange(adata.n_obs * 4, dtype=np.float32).reshape(adata.n_obs, 4)
    # npz stores a *superset* order (e.g. raw cells); provide obs_names
    keep = np.arange(adata.n_obs)[::2]
    p = tmp_path / "emb.npz"
    np.savez(p, emb=emb, obs_names=adata.obs_names.astype(str))
    out = load_input_emb(p, adata[keep])
    assert out.shape[0] == len(keep)
    assert np.array_equal(out, emb[keep])


def test_load_input_emb_misaligned_fails(tmp_path):
    adata = make_adata()
    emb = np.zeros((adata.n_obs, 4), dtype=np.float32)
    p = tmp_path / "bad.npz"
    np.savez(p, emb=emb,
             donor=np.array(["zzz"] * adata.n_obs),
             cell_type=adata.obs["cell_type"].astype(str).to_numpy())
    with pytest.raises(ValueError, match="misaligned"):
        load_input_emb(p, adata)
