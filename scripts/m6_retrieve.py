#!/usr/bin/env python3
"""M6 — retrieve grounding text per cell type from the PubMed FTS5 index.

Runs on the cluster (DB on beegfs; login node is fine — sqlite3 only).
For each cell type × arm, runs the query ladder (src/biocellai/retrieval.py),
pools hits, reranks by marker coverage, and writes:

  data/retrieval/m6/{arm}/{cell_type}.jsonl   — top docs (pmid, title, ...)
  data/text/type_desc_pubmed_{arm}.json       — {type: caption} (type_llm contract)
  {outdir}/retrieval_metrics.csv              — recovery-rate stats per type × arm
  {outdir}/report.md

Arms (query side / output side):
  rag_markers        marker queries, name stripped from output text
  rag_markers_alias  + HGNC alias expansion (recall boost)
  rag_name           name-phrase queries, raw output  (label-derived upper bound)
  rag_name_stripped  name queries, name stripped from output

Usage:
    python scripts/m6_retrieve.py --db /beegfs/.../pubmed_full.db \
        --markers data/manifests/markers_train_s0.json \
        --outdir experiments/m6_retrieval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from biocellai.retrieval import (  # noqa: E402
    IMMUNE_CONTEXT,
    NEURAL_CONTEXT,
    docs_to_caption,
    load_aliases,
    retrieve_for_type,
    _connect,
)

ARMS = ["rag_markers", "rag_markers_alias", "rag_name", "rag_name_stripped",
        "rag_curated", "rag_curated_alias"]

CONTEXTS = {"blood": IMMUNE_CONTEXT, "brain": NEURAL_CONTEXT}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--markers", default="data/manifests/markers_train_s0.json")
    p.add_argument("--curated-markers", default="data/manifests/panglao_markers.json",
                   help="label-free curated markers (PanglaoDB) for rag_curated arms")
    p.add_argument("--aliases", default="data/gene_aliases.json")
    p.add_argument("--outdir", default="experiments/m6_retrieval")
    p.add_argument("--textdir", default="data/text")
    p.add_argument("--docdir", default="data/retrieval/m6")
    p.add_argument("--per-query-k", type=int, default=40)
    p.add_argument("--pool-k", type=int, default=60)
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--context", choices=list(CONTEXTS), default="blood")
    args = p.parse_args()

    markers = json.loads(Path(args.markers).read_text())
    curated = (json.loads(Path(args.curated_markers).read_text())
               if Path(args.curated_markers).exists() else {})
    aliases = load_aliases(args.aliases)
    conn = _connect(args.db)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.textdir).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for arm in args.arms:
        is_curated = arm.startswith("rag_curated")
        use_name = arm.startswith("rag_name")
        use_alias = arm.endswith("alias")
        strip = not arm == "rag_name"
        arm_markers = curated if is_curated else markers
        if is_curated and not curated:
            print(f"[{arm}] skipped — no curated markers file", flush=True)
            continue
        descs: dict[str, str] = {}
        docdir = Path(args.docdir) / arm
        docdir.mkdir(parents=True, exist_ok=True)

        for ct, ms in arm_markers.items():
            t0 = time.time()
            docs, stats = retrieve_for_type(
                conn, ms,
                cell_type=ct if use_name else None,
                aliases=aliases if use_alias else None,
                per_query_k=args.per_query_k,
                pool_k=args.pool_k,
                context_vocab=CONTEXTS[args.context],
            )
            cap = docs_to_caption(
                docs, max_chars=args.max_chars, strip_type=ct if strip else None
            )
            descs[ct] = cap
            safe = ct.replace("/", "_").replace(" ", "_")
            with open(docdir / f"{safe}.jsonl", "w") as f:
                for d in docs:
                    f.write(json.dumps(
                        {k: d.get(k) for k in
                         ("pmid", "year", "journal", "title", "abstract",
                          "coverage", "score", "hits")}
                    ) + "\n")
            rows.append(dict(
                arm=arm, cell_type=ct, n_markers=len(ms), caption_chars=len(cap),
                elapsed_s=round(time.time() - t0, 1), **stats,
            ))
            print(f"[{arm}] {ct}: {stats['n_pooled']} pooled → "
                  f"{stats['n_returned']} docs, cov≥.5={stats['n_cov_ge_half']}, "
                  f"cap={len(cap)}ch ({stats['n_queries']}q, {rows[-1]['elapsed_s']}s)",
                  flush=True)

        Path(args.textdir, f"type_desc_pubmed_{arm}.json").write_text(
            json.dumps(descs, indent=2))

    mf = outdir / "retrieval_metrics.csv"
    with open(mf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# M6 — literature retrieval for external grounding", "",
             "| arm | type | pooled | returned | cov≥0.5 | mean_cov | caption chars |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['cell_type']} | {r['n_pooled']} | "
            f"{r['n_returned']} | {r['n_cov_ge_half']} | {r['mean_cov']:.2f} | "
            f"{r['caption_chars']} |")
    (outdir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(rows)} type×arm rows → {mf}")


if __name__ == "__main__":
    main()
