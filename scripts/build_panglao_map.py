#!/usr/bin/env python3
"""Build the label-free curated marker map from PanglaoDB → JSON manifest.

PanglaoDB markers are EXTERNAL knowledge: curated from published scRNA-seq
literature, not derived from our train-donor labels. This is the fully
label-free grounding source for M6.

Usage:
    python scripts/build_panglao_map.py \
        --tsv data/panglao_markers.tsv.gz \
        --out data/manifests/panglao_markers.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

# Tabula Sapiens Cell Ontology names → PanglaoDB cell-type names.
# PanglaoDB lacks CD4/CD8 and monocyte-subtype resolution; honest mapping.
PANGLAO_MAP = {
    "B cell": ["B cells"],
    "CD4-positive, alpha-beta T cell": ["T cells"],
    "CD8-positive, alpha-beta T cell": ["T cells"],
    "classical monocyte": ["Monocytes"],
    "erythrocyte": ["Erythroid-like and erythroid precursor cells",
                    "Erythroblasts"],
    "hematopoietic precursor cell": ["Hematopoietic stem cells"],
    "intermediate monocyte": ["Monocytes"],
    "macrophage": ["Macrophages"],
    "mature NK T cell": ["Natural killer T cells"],
    "monocyte": ["Monocytes"],
    "naive thymus-derived CD4-positive, alpha-beta T cell":
        ["T cells naive", "T cells"],
    "natural killer cell": ["NK cells"],
    "neutrophil": ["Neutrophils"],
    "non-classical monocyte": ["Monocytes"],
    "plasma cell": ["Plasma cells"],
    "platelet": ["Platelets", "Megakaryocytes"],
}


# SEA-AD cortical Subclass → PanglaoDB names. PanglaoDB resolves only
# coarse neural categories — the honest mapping shows where curated
# knowledge thins: all glutamatergic subclasses collapse to
# "Glutaminergic neurons", all interneurons to "GABAergic neurons".
# Subclass-level discrimination must come from literature retrieval.
_GLUT = ["Glutaminergic neurons", "Neurons"]
_GABA = ["GABAergic neurons", "Interneurons"]
BRAIN_MAP = {
    # glutamatergic cortical subclasses
    **{s: _GLUT for s in ("L2/3 IT", "L4 IT", "L5 IT", "L6 IT",
                          "L6 IT Car3", "L5/6 NP", "L6 CT", "L6b",
                          "L5 ET")},
    # GABAergic cortical subclasses
    **{s: _GABA for s in ("Chandelier", "Lamp5", "Lamp5 Lhx6", "Pax6",
                          "Pvalb", "Sncg", "Sst", "Sst Chodl", "Vip")},
    # non-neuronal
    "Astrocyte": ["Astrocytes"],
    "Endothelial": ["Endothelial cells (blood brain barrier)",
                    "Endothelial cells"],
    "Microglia-PVM": ["Microglia"],
    "Oligodendrocyte": ["Oligodendrocytes"],
    "OPC": ["Oligodendrocyte progenitor cells"],
    "VLMC": ["Fibroblasts"],  # VLMCs are fibroblast-like mural cells
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--map", choices=["blood", "brain"], default="blood")
    args = p.parse_args()
    cell_map = PANGLAO_MAP if args.map == "blood" else BRAIN_MAP

    # cell type -> {gene: best sensitivity_human} (human, canonical only)
    scores: dict[str, dict[str, float]] = {}
    with gzip.open(args.tsv, "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if "Hs" not in row["species"].split():
                continue
            if row["canonical marker"].strip() != "1":
                continue
            ct = row["cell type"].strip()
            g = row["official gene symbol"].strip()
            try:
                s = float(row["sensitivity_human"])
            except ValueError:
                s = 0.0
            cur = scores.setdefault(ct, {})
            cur[g] = max(cur.get(g, 0.0), s)

    out = {}
    for co_name, pd_names in cell_map.items():
        genes: dict[str, float] = {}
        for pn in pd_names:
            for g, s in scores.get(pn, {}).items():
                genes[g] = max(genes.get(g, 0.0), s)
        ranked = sorted(genes, key=lambda g: -genes[g])
        out[co_name] = ranked[:args.top]

    Path(args.out).write_text(json.dumps(out, indent=2))
    empty = [k for k, v in out.items() if not v]
    print(f"{len(out)} types mapped, {len(empty)} empty")
    for k, v in out.items():
        print(f"  {k}: {v[:6]}{'...' if len(v) > 6 else ''}")


if __name__ == "__main__":
    main()
