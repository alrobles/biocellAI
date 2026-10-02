# Literature map — language-grounded cell representations

Working document for the biocellai-devel research line. Organized by *what each
paper teaches us about the CellOLMo question*, with lab-asset hooks.

## 1. What CellOLMo actually wants (from the job posting + Ai2 context)

> "whether grounding a model in biological language and knowledge produces
> measurable improvements over models trained only on molecular data"

Decomposed, the role asks for:

1. **Reproduce & evaluate** scGPT, Geneformer, CellWhisperer, C2S-Scale on
   brain data — honest evaluation, not adoption.
2. **Controlled experiments**: language grounding vs cell-only/text-free
   baselines. ← *This is exactly our M1–M4 harness.*
3. **Multi-level representations**: cell → brain region → donor → pathology
   → disease progression. ← *An axis we have not touched: aggregation.*
4. **SEA-AD specifically**: 84 donors across the AD spectrum (+5
   neurotypical refs), ~2.78M nuclei (MTG/A9) under the ABC Atlas
   taxonomy, ~6M more cells across 11 regions. Disease is *continuous*
   (pathology burden), not discrete types — a different eval regime
   (trajectory/regression rather than classification).
5. **Open release**: code, data, weights — OLMo ethos.

## 2. Landscape

### 2a. Expression-only single-cell foundation models

| Paper | Claim | Critical note |
|-------|-------|---------------|
| scGPT (Cui et al. 2024, Nat Methods) | generative scFM, cell embeddings | zero-shot < HVG+Harmony+scVI baselines (Kedzierska 2025) |
| Geneformer (Theodoris et al. 2023, Nature) | rank-token transformer, transfer learning | same; also < Seurat v5 (bioRxiv 2025.06) |
| scFoundation (Hao et al. 2024, Nat Methods) | 100M-cell pretraining | — |
| UCE (Rosen et al. 2023, bioRxiv) | universal embedding, cross-species, no labels | closest to "label-free transferable space" — protein-embedding-based gene tokens |

### 2b. Language/text-grounded cell models

| Paper | Mechanism | Our read |
|-------|-----------|----------|
| **LangCell** (Zhao et al. 2024, ICML) | contrastive cell–text pretraining on metadata-derived captions | Closest cousin to our harness; captions are label-enriched by construction — our decomposition shows how much of their gain is label-carried |
| **CellWhisperer** (Schaefer et al. 2025, Nat Biotech) | CLIP on 1M cells + AI-curated descriptions, chat LLM | Scale version of our approach; descriptions AI-generated (c.f. our type_llm) |
| **C2S-Scale** (Rizvi et al. 2025, bioRxiv) | cell sentences inside 27B Gemma | text-as-tokenization, not grounding per se; still expression-derived text |
| Cell2Sentence (Levine et al. 2024, ICML) | original cell sentences | same |

### 2c. Knowledge-grounded approaches (the external-KB direction)

| Paper | Source of knowledge | Relevance to M6 |
|-------|--------------------|-----------------|
| **GenePT** (Chen & Zou 2024) | GPT embeddings of NCBI gene summaries | gene-level text prior |
| **scGenePT** (Istrate et al. 2024, CZI) | NCBI/UniProt/GO gene embeddings injected into scGPT | **Key finding: text alone < expression-learned reps, but additive/complementary** — predicts our label-free gap |
| **scCello** (Yuan & Zhan 2024, NeurIPS spotlight) | Cell Ontology graph → alignment loss in pretraining | ontology-as-structure, not text |
| OnClass (Sheng Wang et al. 2021, Nat Commun) | CL graph → unseen-type classification + marker inference | the original "label-free via ontology" |
| KCFM (nudt) | CL knowledge graph → PubMedBERT type embeddings | KG+LM hybrid |
| scHilda (2025, PLoS CompBio) | LLM + KG arbitration for annotation | constrains hallucination via KG |
| **scRAG** (ACL Findings 2025) | RAG over KG triples + similar cells for annotation | closest to our planned genominer arm — but annotation-only, not representation grounding |

### 2d. Critical evaluation literature (why our framing matters)

- **Kedzierska et al. 2025, Genome Biology** (10.1186/s13059-025-03574-x):
  scGPT/Geneformer zero-shot embeddings underperform HVG/Harmony/scVI.
  The field's baseline of trust for scFMs is low — grounding claims need
  controlled evidence, which is our exact niche.
- **"Fundamental Limitations…" bioRxiv 2025.06.26.661767**: scFMs lose to
  Seurat v5 on classification even without perturbation.

## 3. Where our results sit

Our decomposition (text = carrier of label-derived structure; label-free
sources recover 48–66%) is *complementary* to all of the above:

- vs LangCell/CellWhisperer: they show grounding works; we show *why*
  (and how much is label-carried vs knowledge-carried).
- vs scGenePT: they found text-as-prior helps at gene level; we find
  caption-level grounding likewise additive but label-free sources still
  short — consistent picture at different granularity.
- vs scCello/KCFM: ontology structure ≠ free text; our matched-register
  finding applies to both.
- vs Kedzierska et al.: same eval philosophy — zero-shot, baselines,
  honest negatives.

## 4. The research line (positioning)

**"Decompose and close the label-free grounding gap."**

Phase 1 (done): controlled benchmark + decomposition → the gap is in
grounding-source information content, not the alignment mechanism.

Phase 2 (M6, ADR-005): label-free external knowledge — Cell Ontology
definitions, PanglaoDB/CellMarker, literature-retrieved text (genominer
PubMed index + RAG server), with matched-register eval. Falsification gate:
≥0.50 zeroshot F1 held-out donors = positive result.

Phase 3 (portfolio): SEA-AD pilot where eval shifts to pathology
progression (continuous) — plus the lab's differentiated angle:
cross-species grounding where ontologies are sparse (UCE-like question,
text-knowledge answer).

### Lab-asset map

| Asset | Role in this line |
|-------|-------------------|
| genominer (36M PubMed index) + knowledgebase/rag | retrieved grounding text — the Phase-2 differentiator nobody else has |
| ancetralembbedings (ESM-2 650M) | gene→protein embeddings as an alternative grounding channel (UCE-style) or knowledge bridge |
| genoaligner (GPU alignment) | ortholog/homolog mapping for the cross-species arm; sequence-level cell-type conservation |
| phylogenyAI / genoml | species-tree scaffolding for cross-species cell atlases |
| nemotron-eco-reasoner | LoRA/post-training recipe for adapting OLMo-class models to bio text |
| deepla spec-first + DiDAL (dLLM) | methodology: falsification gates, verified citations |

## 5. Open questions for the field (our wedge)

1. Does *retrieved* biological text (not label-derived, not
   LLM-from-labels) close the zero-shot gap on held-out donors?
2. Does grounding benefit survive aggregation — do region/donor-level
   representations improve more than cell-level ones? (CellOLMo's
   explicit second question; nobody has published it.)
3. Cross-species: does text carry orthology when expression can't?
