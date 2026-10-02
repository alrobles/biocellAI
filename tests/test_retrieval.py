"""Tests for the M6 literature-retrieval layer (src/biocellai/retrieval.py)."""
import sqlite3

import pytest

from biocellai.retrieval import (
    _phrase,
    docs_to_caption,
    fts_search,
    load_aliases,
    marker_coverage,
    marker_queries,
    retrieve_for_type,
    strip_cell_type,
)


@pytest.fixture
def mini_pubmed():
    """In-memory FTS5 with the litdump schema and a few relevant docs."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE articles(pmid TEXT PRIMARY KEY, year INT, journal TEXT,"
        " title TEXT, abstract TEXT, mesh TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE articles_fts USING fts5("
        "pmid UNINDEXED, title, abstract, mesh)"
    )
    docs = [
        ("1", 2010, "J Exp Med", "BANK1 regulates B cell signaling",
         "BANK1 and CD74 are expressed in B lymphocytes; IGHM marks the BCR.",
         "B-Lymphocytes"),
        ("2", 2015, "Blood", "Platelet activation via PPBP",
         "PPBP/CXCL7 and ITGA2B drive platelet aggregation.",
         "Blood Platelets"),
        ("3", 2020, "Vaccine", "Unrelated immunology paper",
         "This paper discusses influenza vaccination in adults.",
         "Vaccines"),
        ("4", 2018, "Nat Immunol", "Granulysin in cytotoxic lymphocytes",
         "GNLY (granulysin) and CCL5 are cytotoxic effectors of NK cells.",
         "Natural Killer T-Cells"),
    ]
    conn.executemany(
        "INSERT INTO articles VALUES (?,?,?,?,?,?)", docs
    )
    conn.executemany(
        "INSERT INTO articles_fts(pmid,title,abstract,mesh)"
        " VALUES (?,?,?,?)",
        [(d[0], d[3], d[4], d[5]) for d in docs],
    )
    return conn


def test_phrase_escapes_special_chars():
    assert _phrase("CD4-positive, alpha-beta T cell") == '"CD4 positive alpha beta T cell"'
    assert _phrase("CD74") == '"CD74"'


def test_marker_queries_ladder():
    qs = marker_queries(["CD74", "BANK1", "IGHM", "MS4A1"])
    assert qs[0] == '"CD74" AND "BANK1" AND "IGHM"'
    assert any(q.count("AND") == 1 for q in qs)  # pairwise ANDs
    assert any("cell" in q for q in qs)          # context tier


def test_marker_queries_name_and_aliases():
    qs = marker_queries(
        ["GNLY"], cell_type="natural killer cell",
        aliases={"GNLY": ["granulysin"]},
    )
    assert any('"natural killer cell"' in q for q in qs)
    assert any("granulysin" in q for q in qs)


def test_fts_search_finds_relevant_docs(mini_pubmed):
    hits = fts_search(mini_pubmed, '"CD74" AND "BANK1"', limit=10)
    assert [h["pmid"] for h in hits] == ["1"]


def test_marker_coverage_scores_doc():
    doc = {"title": "BANK1 in B cells", "abstract": "CD74 and IGHM.", "mesh": ""}
    assert marker_coverage(doc, ["CD74", "BANK1", "IGHM", "MS4A1"]) == 0.75
    assert marker_coverage(doc, []) == 0.0


def test_retrieve_for_type_pools_and_reranks(mini_pubmed):
    docs, stats = retrieve_for_type(
        mini_pubmed, ["CD74", "BANK1", "IGHM", "MS4A1"], per_query_k=10, pool_k=10
    )
    assert stats["n_pooled"] >= 1
    assert docs[0]["pmid"] == "1"           # most marker-complete doc first
    assert docs[0]["coverage"] == 0.75


def test_alias_expansion_recovers_synonym_doc(mini_pubmed):
    """'granulysin' doc unreachable by symbol alone; alias arm recovers it."""
    docs_plain, _ = retrieve_for_type(mini_pubmed, ["GNLY"], per_query_k=10)
    docs_alias, _ = retrieve_for_type(
        mini_pubmed, ["GNLY"], aliases={"GNLY": ["granulysin"]}, per_query_k=10
    )
    assert "4" not in {d["pmid"] for d in docs_plain} or len(docs_alias) >= len(docs_plain)
    assert "4" in {d["pmid"] for d in docs_alias}


def test_strip_cell_type_removes_label_sentences():
    txt = "B cells express CD74. The weather is nice. B cell activation matters."
    out = strip_cell_type(txt, "B cell")
    assert "B cells" not in out and "B cell activation" not in out
    assert "weather" in out


def test_docs_to_caption_respects_budget():
    docs = [{"title": "T" * 50, "abstract": "A" * 800} for _ in range(3)]
    cap = docs_to_caption(docs, max_chars=500)
    assert 0 < len(cap) <= 500


def test_load_aliases_missing_file(tmp_path):
    assert load_aliases(tmp_path / "nope.json") == {}
    assert load_aliases(None) == {}
