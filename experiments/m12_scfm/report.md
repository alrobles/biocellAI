# M12 — Scale the cell encoder to a pretrained scFM backbone

Preregistered plan (ADR-009): replace the 2000→256→128→64 MLP with the
scGPT whole-human checkpoint (12L/512d/8H, 33M-cell pretraining) in two
stages — **S1** frozen scGPT features as input to the existing dual-head
contrastive pipeline; **S2** end-to-end fine-tune of the backbone with
the same objective (full-FT and last-2-block PEFT). Gates: improvement
requires Δρ ≥ +0.05 vs the C1 winner (`dual_cog_rag`, 0.604/0.607
MiniLM/SapBERT) on the same donor split, surviving the shuffle null;
S3 ROSMAP transfer conditional on a passing winner.

Runs: `experiments/m12_scgptemb_dual_cog_rag{,_shuf}` (S1, 3 seeds ×
2 text encoders × real/null), `experiments/m12_s2/` (S2, 18 array tasks
= {full,lastN} × {3 seeds} × {MiniLM,SapBERT} + SapBERT shuffles), Slurm
jobs 30360696 (S1) and 30360676 (S2), all COMPLETED.

## Headline: donor CPS_Global ρ (ridge on donor means, held-out donors)

mean over seeds 0/1/2:

| arm | MiniLM | SapBERT | shuf (SapBERT) |
|---|---|---|---|
| S1 frozen-scGPT + dual-head | 0.586 | 0.580 | 0.543 (0.466 MiniLM) |
| S2 full-FT | 0.543 | 0.523 | 0.477 |
| S2 last-2 PEFT | 0.528 | 0.543 | 0.468 |
| **C1 winner (MLP from scratch)** | **0.604** | **0.607** | ~0.2–0.4 (M8/M11 nulls) |
| G2 ridge on raw frozen scGPT | 0.43–0.51 | | |

Best M12 arm = 0.586 → **Δ ≈ −0.02 vs winner → G-M12a FAILS**; no arm
passes → **S3 not run**.

## Where the signal lives (G-M12b)

The diagnostic result of the milestone: **the M12 shuffle nulls do not
collapse**. On the MLP encoder, permuting donor→pathology captions drops
ρ by ~0.2–0.35; on scGPT features the null retains ρ ≈ 0.47–0.54 — most
of the real-arm performance. Real − null gap is only +0.05 to +0.12
across all M12 arms. Reading: the donor-pathology axis is largely
*intrinsic* to scGPT whole-human features (already visible in G2: plain
ridge = 0.43–0.51 with zero task training). The dual-head objective adds
a modest amount on top (+0.07–0.15 over frozen-input ridge), whether the
backbone is frozen (S1), partially (S2b), or fully fine-tuned (S2a).

Per-seed variance is large (seed 2 is systematically harder: 0.25–0.29
across arms); single-seed high point = full-FT MiniLM s0 ρ = 0.753, not
replicated.

## Trajectories and cell type

Ordinal trajectory ρ (mean over seeds/encoders, pooled regions):

| arm | adnc | braak | cerad |
|---|---|---|---|
| S1 real | 0.71 | 0.51 | 0.73 |
| S1 null | 0.31 | 0.34 | 0.21 |
| S2 full-FT | 0.65 | 0.45 | 0.66 |
| S2 last-2 | 0.56 | 0.40 | 0.56 |
| S2 nulls | 0.42–0.48 | 0.29–0.42 | 0.25–0.37 |

Unlike the CPS ridge, the ordinal axes DO separate real from null
(+0.2–0.4) — captions still organise the ordering even where the
donor-mean ridge is saturated by intrinsic feature signal. Full-FT
organises ordinal structure better than last-2 PEFT.

Cell-type grounding preserved end-to-end: zero-shot matched F1 ≈
0.87–0.89, linear probe ≈ 0.98 (S2 slightly above the MLP baseline on
probe — fine-tuning adapts features to the type captions without
damaging them).

## Interpretation

Scaling the molecular encoder ~500× (51M vs ~100k params, plus 33M-cell
pretraining) does not improve donor-level CPS prediction under 84-donor
supervision. The bottleneck is not encoder capacity or pretrained
features — both the frozen (S1) and fine-tuned (S2) variants land at or
below the small grounded MLP, and their nulls show the features already
encode a donor-state axis the captions barely sharpen. Consistent with
the M11 finding that the binding constraint is the *grounding signal*
(caption quality, +0.05–0.13 from PubMed-RAG) rather than the encoder.
Supports the paper claim "objective > scale" — with the caveat that 84
donors may simply underdetermine 51M trainable params (memorisation-
bounded even under PEFT), which is itself a data-scale limit worth
stating.

## Provenance

- Checkpoint `/beegfs/.../data/scgpt_whole_human/best_model.pt`
  (12L/512d/8H, 51 bins, `no_cls` per args.json but `<cls>` prepended at
  tokenisation to match `embed_data`); `use_fast_transformer=False`
  (no flash-attn); bf16 autocast.
- S1 input: `/beegfs/.../data/scgpt_emb_seaad_s0proc.npz` — G2
  embeddings restricted/aligned by `obs_names` to the 278,170
  post-preprocess cells (asserted); S2 retokenises the same cells in-vocab
  (1,980 → 1,542 genes, 77.8%).
- S2 train subsample ≤120k donor-stratified cells/epoch (logged);
  test cells never subsampled. lr 1e-5/3ep (full), 1e-4/5ep (last-2),
  warmup 5%, bs 32, seq ≤1200, official 51-bin DataCollator.
- Metrics: `experiments/m12_scgptemb_*/{metrics,progression}.csv`,
  `experiments/m12_s2/{*/metrics.csv, progression/progression_metrics.csv}`,
  npz per arm in `experiments/m12_s2/embeddings/`.

## Outstanding / caveats

- Seed-2 split is uniformly harder (ρ ~0.25–0.45 across all arms
  including C1) — report means over seeds, not max.
- Region-pooled ρ is the headline; per-region rows in
  `progression_metrics.csv` (66 arm×region entries).
- A single-cell-scale alternative (more donors, not bigger encoder)
  remains the untested scaling axis — motivates next milestone choice.

## Post-hoc diagnostic: composition vs state (m13_composition.py)

To pin down *what* the intrinsic axis is, we evaluated donor-level
cell-type fraction vectors (34 cols, no expression) under the identical
donor splits, vs the arm embeddings and their concatenation
(`experiments/m13_composition/`):

| features | CPS_Global ρ (mean of 3 seeds) |
|---|---|
| **cell-type composition alone** | **0.504** |
| C1 winner emb (MLP grounded) | 0.606 |
| comp + C1 emb | 0.607 |
| S1 frozen-scGPT emb | 0.583 |
| S2 arms emb | 0.52–0.54 |

Composition alone beats pseudobulk-PCA (0.26) and nearly matches the
M12 shuffle nulls (0.47–0.54) — **the intrinsic donor-pathology axis is
largely compositional** (excitatory loss / glial gain shifts every
donor-mean feature space regardless of captions). The grounded
embedding's genuine marginal contribution over composition is
≈ +0.10 ρ on CPS and ≈ 0 on ordinal trajectories (comp PC1 vs
braak/adnc/cerad 0.53–0.67 vs emb 0.51–0.73 ≈ no gain). `comp+emb ≈
emb` confirms embeddings already contain the composition signal.

Consequence for the paper: composition must be a reported baseline in
all donor-level tables; the grounding claim narrows (honestly) to
*within-type state* information beyond compositional shift.
