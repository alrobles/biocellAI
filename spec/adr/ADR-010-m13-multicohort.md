# ADR-010: M13 — multi-cohort training (more donors, not more params)

Status: blocked pending scientific revalidation (ADR-011).

The proposal below is not execution-ready. Shared input space, cohort heads,
caption harmonisation and baselines must be re-specified after v2 revalidation.
No multi-cohort GPU sweep is authorised by this document alone.

## Context

M12's tested scGPT configurations did not improve the historical MLP result.
This does not establish that capacity is irrelevant or that donor diversity is
the binding constraint. The 2000-input MLP has 554,176 encoder parameters.
High shuffle correlations and the composition baseline motivate adjusted
controls, not a causal attribution. Under the historical 84-donor SEA-AD split,
63 donors train and 21 test; these counts must be reconciled with eligibility
and source metadata in ADR-011 before joint training.

Available second cohort (built in M11-A): ROSMAP liu2025
`rosmap_for_seaad.h5ad` = 300,057 nuclei × 2,000 genes (already in the
SEA-AD HVG space), 111 donors, 6 regions, 34 cell types
(Mathys-style subclass nomenclature), 3-level Pathology
(nonAD/earlyAD/lateAD) — no continuous CPS/Braak/CERAD.

M11-A2 recorded ROSMAP pathology-caption transfer to SEA-AD near rho=0.478,
but fitted target-cohort PCA and oriented its sign with target labels. A
strictly source-fitted transfer result is still pending (ADR-011).

**Proposed question:** does joint training improve held-out-donor prediction
relative to a matched single-cohort v2 baseline? The historical cohorts contain
195 donors total, not 195 training donors: approximately 63+83=146 would train
under the 25% held-out rule, subject to the corrected eligibility census.

## Decision

Use the MLP as a tractable testbed; the role of capacity remains unresolved.
Joint train over `concat(seaad_s0, rosmap_for_seaad)` in the shared
2,000-gene space with a `cohort` obs column, donor-held-out on both
cohorts simultaneously (SEA-AD splits seeds 0/1/2 unchanged; ROSMAP
donor split with the same seeds and 25% test fraction).

### Caption harmonisation (main design risk)

The cohorts' cell-type vocabularies do not match (SEA-AD ~30
subclasses: "L2/3 IT", "Pvalb", "Sst"; ROSMAP 34 types: "Exc L2-3 IT",
"Inh VIP"). If each cohort kept its own caption table, the type head
would learn *cohort identity* rather than cell type. Decision: build
one **shared caption map** over a unified label space:

- Map both cohorts' labels onto a canonical subclass vocabulary
  (manual alias table, e.g. "Exc L2-3 IT" → "L2/3 IT", "Mic" →
  "Microglia", "Ast" → "Astro"); types with no confident counterpart
  fall back to their class-level caption ("an inhibitory neuron").
- Ship `data/text/m13/celltype_alias_map.json` + the merged caption
  file; the mapping is provenance-logged and frozen before training.

### Pathology heads

Per-cohort pathology captions (different supervision quality):
- SEA-AD: existing `dual_cog_rag` pathology head (PubMed-RAG captions
  keyed on cognitive status).
- ROSMAP: 3-level pathology captions (new `m13/pathology_rosmap_level`
  caption set: nonAD/earlyAD/lateAD with published descriptions).

Same dual-head loss; cells only contribute to their cohort's pathology
head (cohort-masked InfoNCE so that a ROSMAP "nonAD" cell is never
pulled toward a SEA-AD "severe cognitive decline" caption — coarse and
continuous targets must not alias).

### Arms (seeds 0/1/2 × MiniLM/SapBERT where stated)

| arm | train data | pathology heads | purpose |
|-----|-----------|-----------------|---------|
| J0 | SEA-AD only | cog_rag | = C1 winner (re-run for the joint protocol) |
| J1 | SEA-AD + ROSMAP | both heads | donor diversity + pathology |
| J2 | SEA-AD + ROSMAP | SEA-AD only | isolates "more type-grounded cells" from "more pathology donors" |
| J3 | J1 with ROSMAP pathology shuffled | ROSMAP head null | specificity of ROSMAP supervision |
| J4 | J1, SEA-AD donor→label shuffled | SEA-AD null | the standard null on the eval cohort |

Subsampling: none needed for the MLP (≈578k cells/epoch is cheap);
batch composition is stratified ~50/50 by cohort per batch to keep both
heads active.

## Evaluation

- **Primary (unchanged)**: SEA-AD donor CPS_Global ρ on the same
  held-out donors per seed → comparable to every milestone since M7.
- ROSMAP held-out donors: path_level ρ (ridge + ordinal trajectory) —
  same protocol as M11-A1.
- Cross-cohort transfer re-run post-joint (both directions).
- Composition diagnostic (m13_composition) on the joint embeddings to
  keep the intrinsic-axis confound honest.

## Gates (pre-registered)

| claim | gate |
|---|---|
| G-M13a donor diversity helps | best joint arm Δρ ≥ +0.05 vs J0 on SEA-AD held-out, surviving J4 |
| G-M13b attribution | J1 > J2 → pathology donors drive it; J1 ≈ J2 → it's cell-type/representation diversity (still useful, weaker claim) |
| G-M13c ROSMAP co-benefit | joint model ≥ M11-A1 within-ROSMAP ρ (0.51) |
| honest reporting | cohort-masked heads; alias map + caption files frozen in git; identical SEA-AD splits |

## Risks

- **Cohort confound**: type head could learn cohort ID. Mitigation:
  shared caption space + explicit check that held-out-donor cell-type
  zero-shot doesn't degrade vs J0; report cohort-ID probe as a
  diagnostic row (a model that encodes cohort should not get CPS lift).
- **Coarse ROSMAP pathology** dilutes the pathology head → J2 arm
  isolates it; if J1 < J2 drop the ROSMAP pathology head.
- **Label leakage via region**: HC/EC/TH/AG exist only in ROSMAP;
  SEA-AD regions are the eval targets — region is in captions only via
  cell type, no explicit region head.
