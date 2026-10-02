# ADR-009: M12 — scale the cell encoder to a pretrained scFM backbone

Status: decided (2026-09-25) — outcome: **negative for scale**; see
`experiments/m12_scfm/report.md`. No arm passed G-M12a (best = 0.586 vs
0.604/0.607 winner); shuffle nulls retained ρ ≈ 0.47–0.54 → the
donor-pathology axis is largely intrinsic to frozen scGPT features.
S3 transfer not triggered (no winner).

## Context

Current encoder (ADR-002, unchanged M1→M11): `CellEncoder` = MLP
2000→256→128→64, trained from scratch on log-normalised HVG expression.
The M11 winner `dual_cog_rag` (MLP + PubMed-retrieved pathology captions)
reaches donor CPS_Global ρ = 0.604/0.607 (MiniLM/SapBERT).

G2 measured external scFMs on the identical protocol, frozen, no
grounding: **scGPT whole-human ρ = 0.43–0.51**, Geneformer V2 0.13–0.46,
CellWhisperer-CLIP 0.34–0.58. Two facts motivate M12:

1. scGPT features already carry ~half the donor-pathology signal with
   zero task training → the 33M-cell pretraining prior sees the axis.
2. Our grounded MLP beats every frozen scFM → the contrastive objective
   adds what pretraining lacks. The open question is whether the two
   compose: does grounding a *pretrained* backbone beat both?

Priors against naïve optimism: 84 donors of supervision vs ~51M
trainable params → memorisation is the main risk; and S1 may already
saturate the scFM features' linear content (ridge 0.51).

**Corrected pipeline fact (discovered during M12 design):** `load_dataset`
re-runs `preprocess` on the pre-built s0 h5ad, so the effective
train/eval set for all M8–M11 arms is **278,170 cells × 1,980 genes**
(post-filter), not the 440,390×2,000 stored on disk. The G2 scGPT
embeddings (440,390 rows) therefore do NOT row-align with the arm
inputs; M12 must align by `obs_names` (S1) or re-tokenise the same
post-preprocess cells (S2). Of the 1,980 processed genes, ~1,530 are in
the scGPT vocab (77%).

## Decision

Two sequential stages on the identical protocol — same s0 h5ad, same
donor splits (seeds 0/1/2), same `dual_cog_rag` recipe (type head:
`m7/type_desc_pubmed_rag_markers_alias_allregions.json`; pathology head:
`m11/pathology_cognitive_rag.json`), same donor-pool eval.

### S1 — frozen scGPT features as encoder input (cheap ablation)

Replace the expression matrix with the precomputed G2 scGPT embeddings
(512-d), restricted and aligned to the post-preprocess cells via
`obs_names`. Everything else identical: MLP(512→256→128→64) + dual
heads + InfoNCE. This isolates *grounding on frozen scFM features*:
- vs ridge-on-scGPT (0.43–0.51) → what does the contrastive objective
  add to frozen features?
- vs expression-MLP (0.604) → are scFM features even better input?

### S2 — end-to-end scGPT fine-tune with the dual-head objective

`TransformerModel` whole-human checkpoint (12L/512d/8H, binned input,
`pad_value=-2`, 51 bins), forward → `model._encode` → `<cls>` token
embedding (position 0, matching `embed_data`), → `proj_t`/`proj_p`
heads → the same dual-InfoNCE loss. Input = post-preprocess cells,
per-cell nonzero genes (in-vocab only), `<cls>` prepended, official
`DataCollator(do_binning=True, max_length=1200, sampling=True,
keep_first_n_tokens=1)` — identical tokenisation as G2.

Fine-tuning arms (preregistered, chosen to bracket the memorisation
risk):

| arm | trainable params | lr | epochs |
|-----|------------------|----|--------|
| S2a full-FT | all (~51M) | 1e-5 | 3 |
| S2b last-2  | last 2 encoder blocks only | 1e-4 | 5 |
| S2-shuf    | winner config, donor→pathology permuted | as winner | — |

No flash-attn on the cluster venv → standard `nn.TransformerEncoder`
path (`use_fast_transformer=False`; `load_pretrained` remaps
`Wqkv→in_proj`). Throughput bound: donor-stratified subsample to
≤120k train cells per epoch if the measured epoch exceeds ~45 min on
A100; the cap and seed of the subsample are logged. Test cells are
never subsampled.

### S3 (conditional) — winner → ROSMAP A2 transfer

Repeat the M11-A2 eval with the scaled encoder: encode ROSMAP s0 cells
through the fine-tuned checkpoint (gene symbols → scGPT vocab directly,
no var_names aligner needed), donor-pool, trajectory/ridge on
`path_level`. Gate: ρ ≥ 0.35 (same as A2).

## Evaluation (identical to M7–M11)

- Save `emb_s{seed}_{arm}_{tex}.npz` (emb = pathology-head projection,
  emb_type = type-head, donor/cell_type/region/is_test) →
  `scripts/m7_progression.py` computes donor CPS_Global ρ (ridge,
  held-out donors), adnc/cerad/braak trajectories, per-region rows.
- Cell-type zero-shot matched + linear probe on the type head.
- Shuffle-null arm under the same split; bootstrap CIs on headline ρ.

## Gates / success criteria (pre-registered)

| claim | gate |
|---|---|
| G-M12a scale helps | best S2 arm Δρ ≥ +0.05 vs C1 winner (0.604/0.607) same split AND survives shuffle-null |
| G-M12b where the gain lives | S2 > S1 → fine-tuning; S1 ≈ S2 → frozen features saturate; both ≤ C1 → backbone scale is not the binding constraint (honest negative: strengthens the "objective > scale" claim) |
| G-M12c transfer | winner → ROSMAP ρ ≥ 0.35 |
| honest reporting | identical splits; vocab/args.json + subsample seeds logged per run |

## Risks

- **Donor memorisation** (84 donors, ~51M params): PEFT arm S2b +
  donor-held-out eval + shuffle null bound this.
- **Gene coverage**: 23% of HVGs dropped at vocab match (same as G2 —
  consistent comparison, documented).
- **Throughput without flash-attn**: seq ≤1200, bs 32, fp16 autocast;
  if a full epoch exceeds ~45 min on A100, donor-stratified cap to
  120k cells/epoch (logged; preregistered above).
- **Stochastic truncation at eval**: `sampling=True` drops >1200-token
  genes at random — matches G2 protocol; eval seeded per seed.
- **Numerical parity with G2**: S2 uses `use_fast_transformer=False`
  (G2 embeddings were produced with the fast path requested but no
  flash-attn installed — a smoke check verifies checkpoint load +
  embedding sanity before the sweep).
