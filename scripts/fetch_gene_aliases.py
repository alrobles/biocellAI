#!/usr/bin/env python3
"""Fetch HGNC aliases for marker genes → data/gene_aliases.json.

Query-expansion map for M6 retrieval: symbol → [approved name, alias
symbols, alias names]. Runs on the login node (needs internet to reach
genenames.org); produces a small static JSON consumed by m6_retrieve.py.

Usage:
    python scripts/fetch_gene_aliases.py \
        --markers data/manifests/markers_train_s0.json \
        --out data/gene_aliases.json
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

HGNC_URL = "https://rest.genenames.org/fetch/symbol/{sym}"


def fetch_symbol(sym: str) -> list[str]:
    req = urllib.request.Request(
        HGNC_URL.format(sym=sym), headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            docs = json.loads(r.read())["response"]["docs"]
    except Exception:
        return []
    if not docs:
        return []
    d = docs[0]
    out = []
    for k in ("name", "alias_symbol", "alias_name", "prev_symbol", "prev_name"):
        v = d.get(k)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out.extend(v)
    # dedupe, drop the query symbol itself and overly long phrases
    seen, res = set(), []
    for a in out:
        a = a.strip()
        if a and a.lower() != sym.lower() and len(a) <= 40 and a not in seen:
            seen.add(a)
            res.append(a)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markers", required=True)
    p.add_argument("--out", default="data/gene_aliases.json")
    args = p.parse_args()

    markers = json.loads(Path(args.markers).read_text())
    genes = sorted({g for ms in markers.values() for g in ms})
    aliases = {}
    for i, g in enumerate(genes):
        aliases[g] = fetch_symbol(g)
        print(f"[{i+1}/{len(genes)}] {g}: {len(aliases[g])} aliases", flush=True)
    Path(args.out).write_text(json.dumps(aliases, indent=2))
    n = sum(1 for v in aliases.values() if v)
    print(f"{n}/{len(genes)} genes have aliases → {args.out}")


if __name__ == "__main__":
    main()
