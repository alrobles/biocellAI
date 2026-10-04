#!/usr/bin/env python3
"""Derive the ROSMAP PanglaoDB marker reference for R5-MARKERS.

ROSMAP uses a different cell-type vocabulary than SEA-AD. This maps each
ROSMAP label onto the SAME canonical PanglaoDB marker sets already
extracted for the brain map (`data/manifests/panglao_brain_markers.json`)
plus the T-cell set from `data/manifests/panglao_markers.json`. No new
external data is introduced — provenance is the parent manifests.

ROSMAP types without a PanglaoDB category (CPEC, Epd) get an empty list
and are reported as NOT_EVALUABLE downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# panglao_brain_markers.json stores identical lists per biological
# category; pick a representative key per category.
BRAIN_KEYS = {
    "glut": "L2/3 IT",
    "gaba": "Pvalb",
    "astro": "Astrocyte",
    "endo": "Endothelial",
    "micro": "Microglia-PVM",
    "oligo": "Oligodendrocyte",
    "opc": "OPC",
    "fibro": "VLMC",  # VLMCs are fibroblast-like mural cells (M7 map)
}

ROSMAP_MAP: dict[str, list[str]] = {
    # excitatory -> glutamatergic
    **{t: ["glut"] for t in (
        "Exc L2-3 IT", "Exc L3-4 IT", "Exc L3-5 IT", "Exc L4-5 IT-1",
        "Exc L4-5 IT-2", "Exc L5-6 IT", "Exc L5/6 IT Car3", "Exc L6 IT",
        "Exc L5/6 NP", "Exc L5 ET", "Exc L6 CT", "Exc L6b", "Exc EC",
        "Exc CA pyramidal cells", "Exc DG granule cells", "Exc HC",
        "Exc TH")},
    # inhibitory -> GABAergic
    **{t: ["gaba"] for t in (
        "Inh VIP", "Inh LAMP5", "Inh SST", "Inh PVALB", "Inh PAX6",
        "Inh MEIS2")},
    "Ast": ["astro"],
    "Oli": ["oligo"],
    "OPC": ["opc"],
    "Mic": ["micro"],
    "End": ["endo"],
    "Per": ["fibro"],   # pericytes: mural cells, PanglaoDB fibroblast set
    "SMC": ["fibro"],   # smooth-muscle mural cells, same caveat
    "Fib": ["fibro"],
    "T": ["tcell"],
    # CPEC (choroid-plexus epithelial) and Epd (ependymal): no PanglaoDB
    # category in the extracted manifests -> NOT_EVALUABLE.
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--brain", default="data/manifests/panglao_brain_markers.json")
    p.add_argument("--blood", default="data/manifests/panglao_markers.json")
    p.add_argument("--out", default="data/manifests/panglao_rosmap_markers.json")
    args = p.parse_args()

    brain = json.loads(Path(args.brain).read_text())
    blood = json.loads(Path(args.blood).read_text())
    cat = {k: brain[v] for k, v in BRAIN_KEYS.items()}
    cat["tcell"] = blood["CD4-positive, alpha-beta T cell"]

    out: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for ros_name, cats in ROSMAP_MAP.items():
        genes: dict[str, None] = {}
        for c in cats:
            for g in cat[c]:
                genes.setdefault(g)
        out[ros_name] = list(genes)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{len(out)} ROSMAP types mapped "
          f"({len(unmapped)} unmapped: {unmapped})")
    n_nonempty = sum(bool(v) for v in out.values())
    print(f"{n_nonempty} types with reference markers -> {args.out}")


if __name__ == "__main__":
    main()
