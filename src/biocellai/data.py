"""Data layer: AnnData loading, multi-donor queries, auditable manifests.

Contract: every loader returns an AnnData with
  - .X normalized (log1p + HVG subset applied on a copy)
  - .obs["cell_type"]  categorical labels
  - .obs["donor_id"]   donor identifier for held-out splits
  - .var_names         HVG gene symbols
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetManifest:
    """Auditable record of how a dataset slice was produced."""

    name: str
    source: str
    filters: dict
    n_cells: int
    n_genes_hvg: int
    n_donors: int
    n_cell_types: int
    cells_per_donor: dict
    cells_per_type: dict
    seed: int
    date: str

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


def use_gene_symbols(adata):
    """cellxgene var_names are Ensembl IDs — captions need gene symbols.

    Maps var_names to var['feature_name'] when present; deduplicates.
    Must run BEFORE preprocess/HVG selection and before any caption code —
    otherwise marker captions carry semantically-empty IDs.
    """
    if "feature_name" not in adata.var.columns:
        return adata
    adata = adata.copy()
    adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()
    return adata


def preprocess(adata, n_hvg: int = 2000, seed: int = 0):
    """Standard scRNA preprocessing: QC-light, normalize, log1p, HVG.

    Expects raw counts in .X. Returns a new AnnData restricted to HVGs.
    """
    import scanpy as sc

    validate_counts(adata)
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_hvg, adata.n_vars - 1))
    return adata[:, adata.var.highly_variable].copy()


def load_pbmc3k(n_hvg: int = 2000, seed: int = 0):
    """Smoke-test dataset: pbmc3k_processed ships with louvain cell-type labels.

    X is already log-normalized — we only subset to top-variance genes.
    Single donor, so pseudo-donors are assigned at random purely to exercise
    the held-out codepath. Pipeline verification only, not a real eval.
    """
    import scanpy as sc

    adata = sc.datasets.pbmc3k_processed().copy()
    adata.obs["cell_type"] = adata.obs["louvain"].astype(str)

    # top-variance genes as a cheap HVG stand-in (data already normalized)
    variances = np.asarray(adata.X.var(axis=0)).ravel()
    top = np.argsort(variances)[::-1][: min(n_hvg, adata.n_vars)]
    adata = adata[:, sorted(top)].copy()

    rng = np.random.default_rng(seed)
    adata.obs["donor_id"] = pd.Categorical(rng.choice(["d0", "d1", "d2"], size=adata.n_obs))
    return adata


def query_tabula_blood(
    max_cells: int = 50_000,
    min_cells_per_type: int = 100,
    n_hvg: int = 2000,
    seed: int = 0,
    census_version: str = "2025-11-08",
    tissue_general: str = "blood",
    cache_dir: str | Path | None = "data/raw",
):
    """Tabula Sapiens blood subset via cellxgene_census.

    `collection_name` lives on the dataset table, not obs — so we first resolve
    the Tabula Sapiens dataset_ids, then filter obs by dataset_id +
    tissue_general. Cell types need >= min_cells_per_type support; subsample is
    stratified by donor. The raw download is cached as h5ad under cache_dir.
    Returns (adata, manifest). Needs census network access on cache miss.
    """
    import cellxgene_census

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"tabula_{tissue_general}_{max_cells}_s{seed}.h5ad"

    filters = {
        "collection": "Tabula Sapiens",
        "tissue_general": tissue_general,
        "is_primary_data": True,
        "min_cells_per_type": min_cells_per_type,
        "max_cells": max_cells,
        "census_version": census_version,
    }

    if cache_path is not None and cache_path.exists():
        import anndata

        adata = anndata.read_h5ad(cache_path)
        adata = use_gene_symbols(adata)
    else:
        with cellxgene_census.open_soma(census_version=census_version) as census:
            ds = census["census_info"]["datasets"].read().concat().to_pandas()
            ts_ids = ds.loc[
                ds["collection_name"].str.contains("Tabula Sapiens", case=False, na=False),
                "dataset_id",
            ].tolist()
            ids_sql = "','".join(ts_ids)
            value_filter = (
                f"dataset_id in ['{ids_sql}'] and "
                f"tissue_general == '{tissue_general}' and is_primary_data == True"
            )
            obs = (
                census["census_data"]["homo_sapiens"]
                .obs.read(value_filter=value_filter)
                .concat()
                .to_pandas()
            )

        # keep cell types with enough support
        type_counts = obs["cell_type"].value_counts()
        keep_types = type_counts[type_counts >= min_cells_per_type].index
        obs = obs[obs["cell_type"].isin(keep_types)]

        # stratified subsample by donor to preserve the donor distribution
        rng = np.random.default_rng(seed)
        if len(obs) > max_cells:
            frac = max_cells / len(obs)
            sampled_idx = (
                obs.groupby("donor_id", observed=True, group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=rng), include_groups=False)
                .index
            )
            obs = obs.loc[sampled_idx]

        with cellxgene_census.open_soma(census_version=census_version) as census:
            adata = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                obs_coords=obs["soma_joinid"].to_numpy(),
            )

        adata = use_gene_symbols(adata)
        adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
        adata.uns["filters"] = filters

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(cache_path)

    adata = preprocess(adata, n_hvg=n_hvg, seed=seed)

    manifest = DatasetManifest(
        name=f"tabula_sapiens_{tissue_general}",
        source="cellxgene_census",
        filters=dict(adata.uns.get("filters", filters)) if hasattr(adata, "uns") else filters,
        n_cells=int(adata.n_obs),
        n_genes_hvg=int(adata.n_vars),
        n_donors=int(adata.obs["donor_id"].nunique()),
        n_cell_types=int(adata.obs["cell_type"].nunique()),
        cells_per_donor=adata.obs["donor_id"].value_counts().to_dict(),
        cells_per_type=adata.obs["cell_type"].value_counts().to_dict(),
        seed=seed,
        date=pd.Timestamp.now().isoformat(),
    )
    return adata, manifest


SEAA_DONOR_COLS = {
    "Braak": "braak",
    "CERAD score": "cerad",
    "Thal": "thal",
    "Overall AD neuropathological Change": "adnc",
    "Cognitive Status": "cognitive_status",
    "Last CASI Score": "casi",
    "APOE Genotype": "apoe",
    "Age at Death": "age_at_death",
    "Sex": "sex",
    "Severely Affected Donor": "severely_affected",
}


def _seaad_normalize_obs(obs: pd.DataFrame, region: str) -> pd.DataFrame:
    """Map official SEA-AD obs columns to the pipeline contract.

    cell_type <- Subclass; donor_id <- 'Donor ID'; donor-level
    neuropathology (Braak/CERAD/Thal/ADNC/CASI/APOE) renamed snake_case.
    """
    out = obs.copy()
    out["cell_type"] = out["Subclass"].astype(str)
    out["donor_id"] = out["Donor ID"].astype(str)
    for src, dst in SEAA_DONOR_COLS.items():
        if src in out.columns:
            out[dst] = out[src]
    out["region"] = region
    return out


def _subset_backed_csr(adata, obs, chunk: int = 20000):
    """Read a row subset of a backed CSR matrix without loading it whole.

    anndata's backed fancy-indexing materializes the full CSR (105GB on
    SEA-AD MTG) — this gathers only the selected rows' elements via h5py.
    Rows are returned in ascending file-position order.
    """
    import anndata as ad
    import scipy.sparse as sp

    pos = np.sort(adata.obs_names.get_indexer(obs.index))
    xgrp = adata.file["X"]
    indptr = xgrp["indptr"][:]
    parts_d, parts_i = [], []
    ip_new = np.zeros(len(pos) + 1, dtype=np.int64)
    for s in range(0, len(pos), chunk):
        p = pos[s : s + chunk]
        counts = indptr[p + 1] - indptr[p]
        total = int(counts.sum())
        elem = (np.repeat(indptr[p], counts)
                + np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
        parts_d.append(xgrp["data"][elem])
        parts_i.append(xgrp["indices"][elem])
        ip_new[s + 1 : s + 1 + len(p)] = ip_new[s] + np.cumsum(counts)
    X = sp.csr_matrix(
        (np.concatenate(parts_d), np.concatenate(parts_i), ip_new),
        shape=(len(pos), adata.n_vars))
    names = adata.obs_names[pos]
    sub = ad.AnnData(X=X, obs=obs.loc[names].copy(), var=adata.var.copy())
    if sub.X.dtype != np.float32:  # SEA-AD MTG is float64 — halve memory early
        sub.X = sub.X.astype(np.float32)
    return sub


def load_seaad_region(
    h5ad_path: str | Path,
    region: str,
    cps_csv: str | Path | None = None,
    max_cells: int = 150_000,
    min_cells_per_type: int = 500,
    seed: int = 0,
):
    """Load one SEA-AD region from the official Allen h5ad.

    Reads backed, subsamples stratified by (donor, Subclass), joins Allen's
    Continuous Pseudo-progression Score by (Donor ID, Brain Region).
    Returns (adata, donor_table) — donor_table is one row per donor with
    pathology columns + CPS, for G5a/G5b progression evaluation.
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = _seaad_normalize_obs(adata.obs, region)

    if cps_csv is not None and Path(cps_csv).exists():
        cps = pd.read_csv(cps_csv)
        cps = cps[cps["Brain Region"] == region][
            ["Donor ID", "CPS_Global", "CPS_Global_ABeta",
             "CPS_Global_pTau", "CPS_Local", "CPS_Local_ABeta",
             "CPS_Local_pTau"]]
        idx = obs.index  # merge resets it — preserve obs_names for slicing
        obs = obs.merge(cps, left_on="donor_id", right_on="Donor ID",
                        how="left", suffixes=("", "_cps"))
        obs.index = idx

    type_counts = obs["cell_type"].value_counts()
    obs = obs[obs["cell_type"].isin(type_counts[type_counts >= min_cells_per_type].index)]

    rng = np.random.default_rng(seed)
    if len(obs) > max_cells:
        keep = (
            obs.groupby(["donor_id", "cell_type"], observed=True, group_keys=False)
            .apply(lambda g: g.sample(
                n=max(1, int(round(len(g) * max_cells / len(obs)))),
                random_state=rng), include_groups=False)
            .index
        )
        obs = obs.loc[keep]

    sub = _subset_backed_csr(adata, obs)
    donor_cols = ["donor_id", "braak", "cerad", "thal", "adnc",
                  "cognitive_status", "casi", "apoe", "age_at_death", "sex",
                  "CPS_Global", "CPS_Global_ABeta", "CPS_Global_pTau",
                  "CPS_Local", "CPS_Local_ABeta", "CPS_Local_pTau"]
    donor_table = (
        sub.obs[[c for c in donor_cols if c in sub.obs.columns]]
        .groupby("donor_id").first()
    )
    return sub, donor_table


def donor_split(adata, donor_key: str = "donor_id", test_fraction: float = 0.25, seed: int = 0):
    """Split cells by donor: whole donors go to test, never partial.

    Returns (train_mask, test_mask) boolean arrays aligned with adata.obs.
    """
    _, test_donors = split_donor_ids(adata.obs[donor_key], test_fraction, seed)
    is_test = adata.obs[donor_key].astype(str).isin(test_donors).to_numpy()
    return ~is_test, is_test


def split_donor_ids(donor_ids, test_fraction: float = 0.25, seed: int = 0):
    ids = pd.Series(donor_ids)
    if ids.isna().any() or not 0 < test_fraction < 1:
        raise ValueError("donor IDs must be present and 0 < test_fraction < 1")
    donors = np.array(sorted(ids.astype(str).unique()))
    if len(donors) < 2:
        raise ValueError("at least two donors are required")
    np.random.default_rng(seed).shuffle(donors)
    n_test = min(len(donors) - 1, max(1, int(round(len(donors) * test_fraction))))
    return donors[n_test:], donors[:n_test]


def donors_from_mask(donor_ids, is_test):
    ids = np.asarray(donor_ids).astype(str)
    mask = np.asarray(is_test)
    if ids.ndim != 1 or mask.shape != ids.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("invalid cell-level split mask")
    mask = mask.astype(bool)
    train, test = np.unique(ids[~mask]), np.unique(ids[mask])
    if not len(train) or not len(test) or set(train) & set(test):
        raise ValueError("donor split is empty or has overlap")
    return train, test


def shuffle_training_donor_labels(obs, label_key, train_donors, seed: int):
    ids = obs["donor_id"].astype(str)
    train = set(map(str, train_donors))
    if not train or not train.issubset(set(ids)):
        raise ValueError("training donors missing from observations")
    labels = obs.loc[ids.isin(train), label_key]
    grouped = labels.groupby(ids.loc[labels.index])
    if labels.isna().any() or (grouped.nunique() != 1).any():
        raise ValueError("training labels must be present and donor-constant")
    donor_labels = grouped.first().sort_index()
    permuted = dict(zip(donor_labels.index, np.random.default_rng(seed).permutation(donor_labels)))
    result = obs[label_key].copy()
    mask = ids.isin(train)
    result.loc[mask] = ids.loc[mask].map(permuted)
    return result


def validate_counts(adata):
    from scipy import sparse

    if "log1p" in adata.uns or adata.uns.get("biocellai_data_state", "counts") != "counts":
        raise ValueError("expected raw counts, not previously processed data")
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        raise ValueError("expected integer-valued counts; inspect X/layers provenance")


def prepare_donor_fold(adata, train_donors, test_donors, n_hvg: int = 2000,
                       min_genes: int = 200, min_cells: int = 3):
    import scanpy as sc

    validate_counts(adata)
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("cell and gene identifiers must be unique")
    if adata.obs["donor_id"].isna().any():
        raise ValueError("missing donor IDs")
    train, test = set(map(str, train_donors)), set(map(str, test_donors))
    if not train or not test or train & test:
        raise ValueError("donor split is empty or has overlap")
    if train | test != set(adata.obs["donor_id"].astype(str)):
        raise ValueError("split must cover exactly the input donors")
    if min(n_hvg, min_genes, min_cells) < 1:
        raise ValueError("QC and HVG limits must be positive")
    out = adata.copy()
    sc.pp.filter_cells(out, min_genes=min_genes)
    ids = out.obs["donor_id"].astype(str)
    if set(ids) != train | test:
        raise ValueError("QC removed all cells of a donor; revise eligibility before freezing splits")
    tr = ids.isin(train).to_numpy()
    keep = np.asarray((out.X[tr] > 0).sum(axis=0)).ravel() >= min_cells
    if keep.sum() < 2:
        raise ValueError("fewer than two genes survive train-only QC")
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    out = out[:, keep].copy()
    training = out[tr].copy()
    if n_hvg < training.n_vars:
        sc.pp.highly_variable_genes(training, n_top_genes=n_hvg)
        out = out[:, training.var["highly_variable"].to_numpy()].copy()
    out.obs["is_test"] = ids.isin(test).to_numpy()
    if out.var.index.name in set(out.var.columns):
        out.var.index.name = "gene_id"
    if out.obs.index.name in set(out.obs.columns):
        out.obs.index.name = "cell_id"
    out.uns["biocellai_protocol"] = "v2"
    out.uns["biocellai_data_state"] = "log1p_hvg"
    out.uns["biocellai_train_donors"] = sorted(train)
    out.uns["biocellai_test_donors"] = sorted(test)
    out.uns["biocellai_normalization"] = "full_transcriptome_total_10000_then_log1p"
    out.uns["biocellai_hvg_fit"] = "train_donors_only"
    return out
