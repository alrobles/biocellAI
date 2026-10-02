#!/usr/bin/env python3
"""M7 probe — enumerate SEA-AD in CELLxGENE Census.

Reports: datasets in the collection, obs columns available, pathology
fields present, cell counts per (tissue, donor) for MTG and V1C.
Run on a node with internet access (login node).
"""
import cellxgene_census
import pandas as pd

COLLECTION_DOI = "10.1101/2023.05.08.539485"

with cellxgene_census.open_soma(census_version="latest") as census:
    # find SEA-AD datasets via collection DOI in dataset metadata
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    seaad = ds[ds["collection_doi"].str.contains("539485", na=False)]
    print(f"=== SEA-AD datasets: {len(seaad)}")
    print(seaad[["dataset_id", "dataset_title", "cell_count",
                 "dataset_total_cell_count"]].to_string(max_colwidth=60))

    obs = census["census_data"]["homo_sapiens"].obs
    cols = set(obs.schema.names)
    path_cols = [c for c in cols if any(k in c.lower() for k in
                 ("braak", "cerad", "thal", "adnc", "pathol", "at8",
                  "6e10", "plaque", "tangle", "cognit", "apoe",
                  "dementia", "late", "age_at", "donor", "disease",
                  "tissue", "cell_type", "subclass", "class"))]
    print("\n=== candidate obs columns:")
    for c in sorted(path_cols):
        print(" ", c)

    for did in seaad["dataset_id"].head(4):
        print(f"\n=== obs value_counts for dataset {did}")
        try:
            df = obs.read(
                value_filter=f"dataset_id == '{did}'",
                column_names=[c for c in path_cols if c != "dataset_id"],
            ).concat().to_pandas()
            print("n cells:", len(df))
            for c in df.columns:
                nun = df[c].nunique()
                if nun <= 30:
                    print(f"  {c}: {df[c].value_counts().head(12).to_dict()}")
                else:
                    print(f"  {c}: {nun} unique, e.g. {list(df[c].unique()[:6])}")
        except Exception as e:
            print("  err:", e)
