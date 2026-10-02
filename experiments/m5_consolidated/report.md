# M5 — Consolidation: does language grounding help cell representations?

**Research question (mirror of Ai2 CellOLMo):** does grounding a cell encoder
in biological language improve representations over a molecular-only model,
evaluated on held-out donors?

## The full evidence table

All runs: Tabula Sapiens blood, donor-held-out splits, seeds 0/1/2,
cell encoder MLP 2000→256→128→64, frozen text encoder, symmetric InfoNCE.
`leakage` rows are probes (upper bounds), not honest grounding results.

| exp | arm | grounding source | text enc | leak | F1 | bal-acc |
|-----|-----|------------------|----------|------|-----|---------|
| M1 30k | supervised | — | — | | 0.590 | 0.608 |
| M1 30k | zeroshot | own_genes | MiniLM | | 0.293 | 0.470 |
| M1 30k | zeroshot | own_genes | MiniLM | ✗ | 0.397 | 0.560 |
| **M3 85k** | supervised | — | — | | **0.603** | 0.618 |
| M3 85k | zeroshot | own_genes | MiniLM | | 0.288 | 0.473 |
| M3 85k | zeroshot | own_genes | SapBERT | | 0.261 | 0.446 |
| **M3 85k** | **zeroshot** | **type_markers** | **MiniLM** | | **0.591** | **0.635** |
| M3 85k | zeroshot | type_markers | SapBERT | | 0.578 | 0.620 |
| M4 85k | zeroshot | LLM honest (markers→prose) | SapBERT | | 0.401 | 0.487 |
| M4 85k | zeroshot | LLM honest | MiniLM | | 0.324 | 0.436 |
| M4 85k | zeroshot | LLM labeled | MiniLM | ✗ | 0.575 | 0.625 |
| M4 85k | zeroshot | LLM labeled | SapBERT | ✗ | 0.553 | 0.601 |

(Full per-seed data: `all_metrics.csv`, `summary.csv` in this directory.)

## Answers to the research questions

**RQ1 — does language grounding beat cell-only?**
Conditionally yes. `type_markers` zero-shot (0.591 F1 / 0.635 bal-acc)
**matches the supervised baseline (0.603/0.618) with zero label-trained
weights**, on donors never seen in training. Text alignment suffices to
transfer class structure across donors — the mechanism CellOLMo hypothesizes
works at this scale.

**RQ2 — which grounding source?**
The source dominates everything else:

- per-cell top-genes (`own_genes`): 0.29 — honest but weak; single-cell
  sparsity makes per-cell lists noisy.
- train-derived marker lists (`type_markers`): 0.59 — best, but note these
  are *label-derived*; the arm is best understood as **supervision distilled
  through text**.
- LLM prose inferred from markers (honest, no name): 0.40 — better than
  own_genes, below marker lists; the LLM contributes real biological
  semantics (lineage, function) but register mismatch with marker-list
  class captions caps the zero-shot score.
- label-in-text (any form): 0.40–0.58 — leakage probes, reported only as
  upper bounds.

**RQ3 — does a biomedical text encoder help?**
On gene-list captions: no (SapBERT ≈ MiniLM — entity-linking pretraining
adds nothing to gene tokens). On natural prose: yes (+0.08 F1 zero-shot
over MiniLM). Text-encoder choice matters only once captions are prose.

## What this portfolio demonstrates (for the CellOLMo application)

- Controlled cell-only vs language-grounded comparison with donor-held-out
  evaluation, leakage probes, and multiple seeds — the eval discipline the
  role requires.
- A real negative-to-positive debugging story: Ensembl/positional var_names
  silently corrupted grounding semantics; the fix (feature_name mapping +
  regression tests) is documented and pinned.
- LLM-generated biological text with full provenance (prompts, raw outputs,
  leak flags) — reusable for knowledge-grounded experiments.
- An honest decomposition of *why* grounding helps: it is a carrier of
  class-discriminative structure, not magic.

## Known limitations (stated plainly)

- `type_markers` uses train labels — it proves text can *carry* supervised
  signal, not that unsupervised text grounding solves annotation.
- `own_genes` (the fully label-free arm) remains weak — pure unsupervised
  grounding is not solved here.
- Zero-shot eval is format-sensitive; a matched-format prose eval would
  likely raise the `type_llm` arms.
- One dataset, one tissue, coarse cell types, MLP encoder — no claim of
  generality beyond this scale.
- Marker table = mean-diff rank, not DE statistics.

## Go / No-Go

**GO — with a sharpened question.** The mechanism is proven; the open gap is
*label-free* grounding that still transfers. Two directions, in priority order:

1. **External-knowledge grounding (recommended next).** Replace label-derived
   markers with sources independent of the training labels: Cell Ontology
   definitions, PanglaoDB/CellMarker entries, or LLM descriptions evaluated
   against matched-format class captions. If a label-free grounding source
   approaches supervised performance on held-out donors, that is a genuine
   CellOLMo-relevant result — and it is exactly where the lab's assets
   (genominer's 36M-abstract PubMed index, RAG server) become differentiators:
   grounding text *retrieved* from literature rather than computed from labels.

2. **SEA-AD pilot (aligned showcase).** The same harness ports to the
   Alzheimer's atlas — donor-level splits become donor×region, cell types
   become neuronal subtypes, and pathology progression becomes the eval axis.
   Worth doing once the external-knowledge arm exists; doing it now would
   just replicate the blood result on a harder dataset.

3. **Eco-evo branch (differentiated, lower priority).** Cross-species cell
   annotation where ontologies are sparse — the setting where text grounding
   has the most to give and where this portfolio would be unique. Defer until
   (1) lands.

## Reproducibility checklist

- [x] spec + ADRs (`spec/`), tests (21 passing), smoke dataset (pbmc3k)
- [x] data manifests + marker tables + LLM prompt/output provenance committed
- [x] seeds {0,1,2} throughout; per-seed metrics preserved
- [x] leakage arms explicitly flagged; superseded invalid runs marked, not deleted
- [x] HPC launch scripts + shared-cache design committed (`scripts/slurm/`)
- [ ] model weights not released (encoders are tiny; regenerable in <6 GPU-hours)
