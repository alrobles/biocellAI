# Release plan — weights, data artifacts, and license

Goal: an open-science release consistent with Ai2/Allen Institute conventions,
without redistributing controlled-access data or overstating scientific status.

## Recommended license stack

| Artifact | License | Rationale |
|---|---|---|
| Code (`src/`, `scripts/`, `spec/`) | **Apache-2.0** | Same license Ai2 uses for OLMo and most Ai2 open-source releases; permissive, includes patent grant, standard for research code |
| Project-generated text artifacts (captions, retrieval corpora, manifests, `data/text/`, `data/retrieval/`) | **CC-BY-4.0** | Standard for derived research data on cellxgene/Zenodo; matches Allen Institute data-sharing norms |
| Trained model weights (where releasable) | **Apache-2.0** | Matches OLMo weight releases; note weights are derived artifacts — see provenance limits below |
| Paper / reports | CC-BY-4.0 | Standard for preprints and documentation |

Concretely: `LICENSE` = Apache-2.0, plus a `NOTICE`/README section stating that
caption and manifest artifacts are under CC-BY-4.0 and that third-party data
keep their original terms.

## What can be released vs. what cannot

| Artifact | Release? | Notes |
|---|---|---|
| Code, specs, ADRs, tests | Yes — already public | Apache-2.0 |
| LLM-generated captions + provenance (`data/text/`) | Yes | Our generated text; CC-BY-4.0. Include prompt/provenance JSONs |
| PubMed retrieval corpora (`data/retrieval/`) | Yes, as abstracts/PMID lists | Verify no full-text paywalled content is included — titles/abstracts only |
| Dataset manifests, donor-split lists, gene lists, hashes | Yes | Reproducibility without redistribution |
| Processed fold `.h5ad` files | **Conditional** | Depends on source license (below) |
| Trained cell-encoder checkpoints | **Conditional by training data** | See provenance limits |
| Raw datasets | **No** | Not ours to redistribute |

### Source-data terms (verified 2026-10-03; decisions in `spec/redistribution_manifest.json`)

- **Tabula Sapiens** (cellxgene): CC-BY-4.0 — processed blood folds and
  Tabula-trained weights are redistributable with attribution.
- **SEA-AD** (Allen Brain Cell Atlas): **CC BY-NC 4.0** + Allen Terms of
  Use + citation policy — derivatives may be shared only
  **non-commercially** with Allen attribution. SEA-AD-trained weights
  may therefore NOT be released under Apache-2.0; options are a
  CC BY-NC 4.0 release or withholding (default: withhold, ship
  manifests + recipe). Raw snRNASeq stays behind the AD Knowledge
  Portal regardless.
- **ROSMAP** (AD Knowledge Portal / RADC, controlled access): **do not**
  redistribute raw or processed donor matrices. Release donor IDs/split
  lists and recipes only.
- **Weights trained on ROSMAP**: controlled-access-derived; **never
  released** (DUO does not permit redistribution of derivatives).
- **Third-party model weights** (scGPT, Geneformer, CellWhisperer,
  SapBERT, MiniLM, OLMo, Qwen caches): **never re-hosted**; cite the
  source repository and version.
- **PubMed caption corpus**: verified titles+abstracts only, truncated
  (~1200 chars/class); no paywalled full text.

## Release channels

1. **GitHub Release** `v0.1.0-alpha` ("Foundations") — code + docs snapshot.
   Tag only after LICENSE and README release notes land.
2. **Hugging Face Hub** (`alrobles/biocellai-*` collections) — trained weights +
   caption artifacts. This is where the Ai2 audience expects open models.
3. **Zenodo** — DOI-minted archive of the repo + manifests for citation.

## Sequenced plan

| Step | Depends on | Owner |
|---|---|---|
| 1. Add LICENSE (Apache-2.0) + NOTICE for CC-BY-4.0 artifacts | **Done** — `LICENSE`, `NOTICE` | repo owner |
| 2. Inventory weights, fold files, logs on HPC + hashes | Unblocked — credential rotated 2026-09-26 (`R0-CREDENTIAL` done) | engineering |
| 3. Classify each artifact by source-data terms | step 2 | engineering + owner |
| 4. Package releasable weights with model cards (training data, protocol version, metrics, intended use, limitations) | step 3 | engineering |
| 5. Package caption/manifest corpus with data cards | **Done (first upload)** — `scripts/hf_release.py`, whitelist + secret scan; https://huggingface.co/datasets/alrobles/biocellai-foundations (249 files). Weights pending inventory | engineering |
| 6. Zenodo deposit + DOI; GitHub alpha tag | **Done** — `alrobles/biocellAI` public, `v0.1.0-alpha` tag + release; DOI `10.5281/zenodo.23103004` | engineering |
| 7. Re-run under v2 protocol → release v1.0 with validated weights/results | `R3-MATRIX`…`R8-RELEASE` | research |

**Do not tag a stable release from historical results.** The alpha tag is an
infrastructure release; the credible scientific release follows the v2 matrix.

## Model-card minimums per checkpoint

- Training cohort + donor counts, split seed(s), preprocessing version
- Objective, encoder, embedding dim, parameter count
- Held-out metrics **with protocol label** (`historical` vs `v2`)
- Known limitations + failure modes (e.g., seed-2 split harder; composition
  baseline strength)
- Source-data license chain
