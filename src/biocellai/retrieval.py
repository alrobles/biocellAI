"""Literature retrieval over the PubMed FTS5 index (litdump/pubmed_full.db).

M6 (ADR-005): external-knowledge grounding. For each cell type we issue a
*ladder* of FTS5 queries — from strict marker conjunctions to expanded
synonym/context queries — pool the hits, and rerank by marker coverage.
The goal is a high literature-recovery rate (recall) without contaminating
the grounding text with the held-out labels.

Schema (ecoseek-litdump build):
    articles(pmid TEXT PRIMARY KEY, year INT, journal TEXT,
             title TEXT, abstract TEXT, mesh TEXT)
    articles_fts USING fts5(pmid UNINDEXED, title, abstract, mesh)

stdlib only — runs on the login node; the DB lives on beegfs.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = "/beegfs/a474r867/litdump/pubmed/pubmed_full.db"
PUBMED_DB = os.environ.get("PUBMED_DB", DEFAULT_DB)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Label-free topical prior: immune/blood context vocabulary. Docs about the
# right cell type should mention this context; marker co-occurrence alone
# retrieves off-topic hits (e.g. adipose-tissue arrays sharing a gene).
IMMUNE_CONTEXT = [
    "lymphocyte", "leukocyte", "immune", "immunity", "pbmc",
    "peripheral blood", "bone marrow", "spleen", "cytotoxic",
    "antibody", "myeloid", "hematopoietic", "cytokine", "lineage",
]

# Topical prior for SEA-AD brain subclasses (M7): docs about cortical cell
# types mention neural context, not immune context.
NEURAL_CONTEXT = [
    "neuron", "neuronal", "cortex", "cortical", "brain", "synaptic",
    "synapse", "glia", "glial", "astrocyte", "oligodendrocyte",
    "microglia", "interneuron", "pyramidal", "myelin", "axon",
    "dendrite", "neurotransmitter", "forebrain", "hippocampus",
    "alzheimer", "dementia",
]

# MeSH headings for name-based queries (curated topical index — high precision).
MESH_HINTS = {
    "B cell": "B-Lymphocytes",
    "CD4-positive, alpha-beta T cell": "CD4-Positive T-Lymphocytes",
    "CD8-positive, alpha-beta T cell": "CD8-Positive T-Lymphocytes",
    "classical monocyte": "Monocytes",
    "intermediate monocyte": "Monocytes",
    "non-classical monocyte": "Monocytes",
    "monocyte": "Monocytes",
    "macrophage": "Macrophages",
    "erythrocyte": "Erythrocytes",
    "natural killer cell": "Killer Cells, Natural",
    "mature NK T cell": "Natural Killer T-Cells",
    "neutrophil": "Neutrophils",
    "plasma cell": "Plasma Cells",
    "platelet": "Blood Platelets",
    "hematopoietic precursor cell": "Hematopoietic Stem Cells",
    "naive thymus-derived CD4-positive, alpha-beta T cell":
        "CD4-Positive T-Lymphocytes",
}


def _phrase(term: str) -> str:
    """FTS5-safe quoted phrase; drops chars that break the query syntax."""
    clean = " ".join(_WORD_RE.findall(term))
    return f'"{clean}"' if clean else '""'


def marker_queries(
    markers: list[str],
    cell_type: str | None = None,
    aliases: dict[str, list[str]] | None = None,
    tissue_context: str = "cell",
) -> list[str]:
    """Query ladder for one cell type, ordered specific → broad.

    L1  AND of top-3 marker symbols            (high precision)
    L2  pairwise ANDs of top-5 markers         (co-occurrence)
    L3  marker AND context term                (disambiguation)
    L4  alias-expanded OR queries              (synonym recall boost)
    L5  cell-type name phrase                  (label-derived; only if given)
    """
    qs: list[str] = []
    m = [mk for mk in markers if mk][:8]

    if len(m) >= 3:
        qs.append(" AND ".join(_phrase(g) for g in m[:3]))
    for i in range(min(5, len(m))):
        for j in range(i + 1, min(5, len(m))):
            qs.append(f"{_phrase(m[i])} AND {_phrase(m[j])}")
    for g in m[:5]:
        qs.append(f"{_phrase(g)} AND ({_phrase(tissue_context)} OR cells OR cellular)")

    if aliases:
        # Synonym-expanded OR query — NOT gated by the context term: FTS5 has
        # no stemming, and gating on "cell" would drop docs saying "cells".
        for g in m[:5]:
            alts = [g, *aliases.get(g, [])][:4]
            if len(alts) > 1:
                qs.append(" OR ".join(_phrase(a) for a in alts))

    if cell_type:
        qs.append(_phrase(cell_type))
        qs.append(f"{_phrase(cell_type)} AND {_phrase('marker')}")
        mesh = MESH_HINTS.get(cell_type)
        if mesh:
            qs.append(f"mesh: {_phrase(mesh)}")
            qs.append(f"mesh: {_phrase(mesh)} AND {_phrase(m[0])}" if m else
                      f"mesh: {_phrase(mesh)}")

    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _connect(db_path: str) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=120)
    conn.row_factory = sqlite3.Row
    return conn


class QueryTimeout(Exception):
    pass


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 50,
               timeout_s: float = 30.0) -> list[dict]:
    """One FTS5 MATCH query → [{pmid, year, journal, title, abstract, mesh, rank}].

    year/journal live in `articles`; the FTS stores (pmid,title,abstract,mesh).
    A progress handler aborts queries that blow past `timeout_s` — a generic
    alias (e.g. "protein") can otherwise scan millions of postings.
    """
    import time
    deadline = time.monotonic() + timeout_s
    ticks = [0]

    def _guard():
        ticks[0] += 1
        return 1 if time.monotonic() > deadline else 0

    sql = """
        SELECT f.pmid, a.year, a.journal, f.title, f.abstract, f.mesh, f.rank
        FROM articles_fts f JOIN articles a ON a.pmid = f.pmid
        WHERE articles_fts MATCH ?
        ORDER BY f.rank LIMIT ?
    """
    conn.set_progress_handler(_guard, 50_000)
    try:
        rows = conn.execute(sql, (query, limit)).fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise QueryTimeout(query) from e
        raise
    finally:
        conn.set_progress_handler(None, 0)
    return [dict(r) for r in rows]


def marker_coverage(doc: dict, markers: list[str], aliases: dict[str, list[str]] | None = None) -> float:
    """Fraction of the query's markers (or their aliases) present in the doc."""
    text = " ".join(str(doc.get(k) or "") for k in ("title", "abstract", "mesh")).lower()
    if not markers:
        return 0.0
    hit = 0
    for g in markers:
        terms = [g.lower(), *(a.lower() for a in (aliases or {}).get(g, []))]
        if any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms):
            hit += 1
    return hit / len(markers)


def retrieve_for_type(
    conn: sqlite3.Connection,
    markers: list[str],
    cell_type: str | None = None,
    aliases: dict[str, list[str]] | None = None,
    per_query_k: int = 40,
    pool_k: int = 60,
    context_vocab: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Run the full ladder, pool by pmid, rerank by coverage + aggregate rank.

    Returns (top docs, retrieval stats). Stats include per-level doc counts
    so the recovery rate is auditable per query tier.
    """
    queries = marker_queries(markers, cell_type=cell_type, aliases=aliases)
    pool: dict[str, dict] = {}
    per_level: list[int] = []
    n_timeouts = 0
    for qi, q in enumerate(queries):
        try:
            hits = fts_search(conn, q, limit=per_query_k)
        except QueryTimeout:
            hits = []
            n_timeouts += 1
        per_level.append(len(hits))
        for rank_i, d in enumerate(hits):
            pmid = d["pmid"]
            e = pool.setdefault(pmid, {**d, "hits": 0, "best_rank": 1e9, "queries": []})
            e["hits"] += 1
            e["queries"].append(qi)
            e["best_rank"] = min(e["best_rank"], float(d["rank"]))
            e["score_agg"] = e.get("score_agg", 0.0) + 1.0 / (rank_i + 1.0)

    docs = list(pool.values())
    for d in docs:
        d["coverage"] = marker_coverage(d, markers, aliases)
        d["context"] = marker_coverage(d, context_vocab or IMMUNE_CONTEXT)
        # rerank: marker coverage dominates, immune context breaks ties toward
        # topical docs, then aggregate reciprocal-rank, then best BM25
        d["score"] = (10.0 * d["coverage"] + 2.0 * d["context"]
                      + d.get("score_agg", 0.0) + 0.01 * (-d["best_rank"]))
    docs.sort(key=lambda d: d["score"], reverse=True)
    docs = docs[:pool_k]

    stats = {
        "n_queries": len(queries),
        "n_query_timeouts": n_timeouts,
        "per_level_hits": per_level,
        "n_pooled": len(pool),
        "n_returned": len(docs),
        "n_cov_ge_half": sum(1 for d in docs if d["coverage"] >= 0.5),
        "n_cov_ge_25": sum(1 for d in docs if d["coverage"] >= 0.25),
        "mean_cov": (sum(d["coverage"] for d in docs) / len(docs)) if docs else 0.0,
    }
    return docs, stats


def strip_cell_type(text: str, cell_type: str) -> str:
    r"""Remove literal cell-type mentions from retrieved text (leakage control).

    Matches the full type phrase — e.g. "B cell" → /\bb\s+cells?\b/i — so
    generic sentences containing "cell" survive but "B cells" does not.
    """
    toks = _WORD_RE.findall(cell_type)
    if not toks:
        return text
    pat = re.compile(r"\b" + r"\s+".join(rf"{re.escape(t)}s?" for t in toks) + r"\b",
                     re.IGNORECASE)
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text) if not pat.search(s)]
    return " ".join(kept).strip()


def docs_to_caption(docs: list[dict], max_chars: int = 1200, strip_type: str | None = None) -> str:
    """Concatenate top-doc title+abstract snippets into a grounding caption."""
    parts, used = [], 0
    for d in docs:
        frag = f"{d.get('title') or ''}. {d.get('abstract') or ''}"
        if strip_type:
            frag = strip_cell_type(frag, strip_type)
        frag = frag.strip()
        if not frag:
            continue
        room = max_chars - used
        if room <= 0:
            break
        parts.append(frag[:room])
        used += len(frag[:room])
    return " ".join(parts)


def load_aliases(path: str | os.PathLike | None) -> dict[str, list[str]]:
    """Optional gene-symbol → [full name, alias symbols, alias names] map."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())
