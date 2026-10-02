#!/usr/bin/env python3
"""M11-C1 — retrieval-grounded pathology captions (replaces manual M8 captions).

For each pathology dimension file (data/text/m8/pathology_*.json) and each
ordinal level key (e.g. 'Not AD' … 'High', 'Braak 0' … 'Braak VI'), runs FTS5
queries against the PubMed index built from:

  core terms    — dimension-specific biology (e.g. neurofibrillary tangles)
  severity tier — vocabulary of the level's disease stage (control → severe)

Ordinal position within the dimension's ordered labels is normalised to a
0–3 severity tier, so 2-, 4- and 6-level dimensions share the tier vocab.

Top docs are pooled across the level's queries, reranked by term coverage,
and concatenated via docs_to_caption — the same caption channel the manual
descriptions occupy, so the arm isolates *source of grounding text*
(retrieved literature vs hand-written). PMIDs + per-level stats are kept.

Output:
  data/text/m11/pathology_{dim}_rag.json      {level_key: caption}
  data/retrieval/m11/{dim}/{level}.jsonl      top docs (pmid, title, score)
  {outdir}/report.md + retrieval_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from biocellai.retrieval import (  # noqa: E402
    NEURAL_CONTEXT,
    _connect,
    docs_to_caption,
    fts_search,
    marker_coverage,
)

# severity-tier vocabulary (index = disease stage, low→high)
SEVERITY_TERMS = {
    0: ["cognitively normal", "healthy aging", "non-demented control",
        "neuropathology-free", "normal aging brain"],
    1: ["mild cognitive impairment", "prodromal Alzheimer",
        "early-stage Alzheimer", "incipient pathology"],
    2: ["moderate Alzheimer disease", "established Alzheimer pathology",
        "intermediate neurofibrillary"],
    3: ["severe Alzheimer disease", "advanced Alzheimer pathology",
        "end-stage dementia", "high neuropathologic change"],
}

# dimension-specific core biology terms (label-free descriptors)
CORE_TERMS = {
    "adnc": ["Alzheimer disease neuropathologic change", "amyloid plaque",
             "neurofibrillary tangle", "tau pathology", "neuritic plaque"],
    "braak": ["neurofibrillary tangle", "tau hyperphosphorylation",
              "Braak staging", "transentorhinal cortex", "tau propagation"],
    "cerad": ["neuritic plaque", "CERAD", "senile plaque density",
              "amyloid deposition cortex"],
    "cognitive": ["cognitive decline", "memory impairment",
                  "dementia severity", "cognitive trajectory aging"],
    "cps": ["cortical pathology burden", "amyloid beta cortical",
            "tau cortical", "global neuropathology"],
    "thal": ["Thal phase", "amyloid phase", "beta-amyloid deposition",
             "diffuse plaque", "amyloid spread"],
    "rosmap": ["Alzheimer disease neuropathology", "amyloid plaque",
               "neurofibrillary tangle", "tau pathology",
               "microglial activation Alzheimer"],
}

MAX_CHARS = 1200


def ordered_keys(dim: str, keys: list[str]) -> list[str]:
    """Ordinal sort of level keys via the shared biocellai.ordinal map."""
    from biocellai.ordinal import level_order
    return level_order(keys, dim=dim)


def retrieve_level(conn, terms, per_query_k, pool_k):
    """Pool FTS hits across term queries, rerank by coverage + recip. rank."""
    pool: dict[str, dict] = {}
    n_timeouts = 0
    for q in terms:
        try:
            hits = fts_search(conn, f'"{q}"', limit=per_query_k)
        except Exception:
            hits, n_timeouts = [], n_timeouts + 1
        for rank_i, d in enumerate(hits):
            e = pool.setdefault(d["pmid"], {**d, "hits": 0, "best_rank": 1e9})
            e["hits"] += 1
            e["best_rank"] = min(e["best_rank"], float(d["rank"]))
            e["score_agg"] = e.get("score_agg", 0.0) + 1.0 / (rank_i + 1.0)
    docs = list(pool.values())
    for d in docs:
        d["coverage"] = marker_coverage(d, terms)
        d["context"] = marker_coverage(d, NEURAL_CONTEXT)
        d["score"] = (10.0 * d["coverage"] + 2.0 * d["context"]
                      + d.get("score_agg", 0.0) + 0.01 * (-d["best_rank"]))
    docs.sort(key=lambda d: d["score"], reverse=True)
    return docs[:pool_k], n_timeouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--indir", default="data/text/m8")
    ap.add_argument("--dims", nargs="+",
                    default=["adnc", "braak", "cerad", "cognitive", "cps",
                             "thal"])
    ap.add_argument("--textdir", default="data/text/m11")
    ap.add_argument("--docdir", default="data/retrieval/m11")
    ap.add_argument("--outdir", default="experiments/m11_retrieval")
    ap.add_argument("--per-query-k", type=int, default=25)
    ap.add_argument("--pool-k", type=int, default=20)
    args = ap.parse_args()

    conn = _connect(args.db)
    Path(args.textdir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows = []

    for dim in args.dims:
        src = Path(args.indir) / f"pathology_{dim}.json"
        if not src.exists():
            print(f"[{dim}] no source file {src} — skipped", flush=True)
            continue
        keys = ordered_keys(dim, list(json.loads(src.read_text())))
        n_lv = len(keys)
        core = CORE_TERMS.get(dim, ["Alzheimer disease neuropathology"])
        descs = {}
        for pos, bk in enumerate(keys):
            tier = round(pos / max(1, n_lv - 1) * 3)
            terms = core + SEVERITY_TERMS[tier]
            t0 = time.time()
            docs, nt = retrieve_level(conn, terms, args.per_query_k,
                                      args.pool_k)
            cap = docs_to_caption(docs, max_chars=MAX_CHARS)
            descs[bk] = cap
            dd = Path(args.docdir) / dim
            dd.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w.-]+", "_", bk)
            with open(dd / f"{safe}.jsonl", "w") as f:
                for d in docs:
                    f.write(json.dumps({k: d.get(k) for k in (
                        "pmid", "year", "journal", "title", "abstract",
                        "coverage", "score", "hits")}) + "\n")
            mc = (sum(d["coverage"] for d in docs) / len(docs)) if docs else 0
            rows.append(dict(dim=dim, level_key=bk, tier=tier,
                             n_terms=len(terms), n_docs=len(docs),
                             mean_cov=round(mc, 3), cap_chars=len(cap),
                             timeouts=nt,
                             elapsed_s=round(time.time() - t0, 1)))
            print(f"[{dim}] {bk} (tier{tier}): {len(docs)} docs cov={mc:.2f} "
                  f"cap={len(cap)}ch ({rows[-1]['elapsed_s']}s)", flush=True)

        (Path(args.textdir) / f"pathology_{dim}_rag.json").write_text(
            json.dumps(descs, indent=2))

    if rows:
        with open(Path(args.outdir) / "retrieval_metrics.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        lines = ["# M11-C1 — retrieval-grounded pathology captions", "",
                 "| dim | level | tier | docs | mean_cov | chars |",
                 "|---|---|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['dim']} | {r['level_key']} | {r['tier']} | "
                         f"{r['n_docs']} | {r['mean_cov']:.2f} | "
                         f"{r['cap_chars']} |")
        (Path(args.outdir) / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(rows)} dim×level rows done")


if __name__ == "__main__":
    main()
