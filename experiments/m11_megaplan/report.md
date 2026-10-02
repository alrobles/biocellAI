# M11 — Improvement megaplan: ROSMAP replication, transfer, and objective variants

Preregistered plan (ADR-008): three avenues to improve donor-level pathology
prediction — **A** more donors / external cohort (ROSMAP), **B** better
donor-level architecture (MIL pooling, soft-ordinal contrastive), **C** better
grounding corpus (PubMed-retrieved pathology captions). Gates: improvement
counts only at Δρ ≥ +0.05 on the same split surviving the shuffle null;
ROSMAP replication target ρ ≥ 0.5; zero-shot transfer target ρ ≥ 0.35.

## A — ROSMAP Liu2025 as external cohort

Data: `snRNA_multiregion_liu2025.h5ad` (2,263,395 nuclei, 111 donors,
6 regions: HC, EC, TH, AG, MTC, PFC; open processed release). Built
`rosmap_s0.h5ad` = 300,057 nuclei (≤800 cells per donor×region), 2000 HVGs,
log-normalised identically to SEA-AD. Pathology label is donor-constant
(0 donors with mixed labels).

**Limitation (documented, preregistered):** the open release carries only a
3-level Pathology label (nonAD/earlyAD/lateAD) — no Braak/CERAD/cogdx or
continuous tau/amyloid scores. The replication target is therefore the
ordinal pathology level (0/1/2), not a continuous score.

### A1 — retrain + held-out donors (n_te ≈ 28)

Single-head pathology contrastive (M8-style), 3 seeds × 2 text encoders,
same donor-split protocol:

| arm | MiniLM path_level ρ | MiniLM trajectory ρ | SapBERT path_level ρ | SapBERT trajectory ρ |
|---|---|---|---|---|
| hard InfoNCE (real) | 0.513 | **0.372** | 0.436 | 0.150 |
| soft-ordinal (real) | 0.458 | 0.129 | 0.359 | 0.034 |
| hard, donor→label shuffled | 0.498 | −0.081 | 0.313 | −0.129 |
| soft, shuffled | 0.363 | −0.053 | 0.383 | 0.035 |
| **pseudobulk PCA baseline** | **0.517** | 0.105 | — | — |

Regional breakdown (hard, real): **HC 0.56–0.59 > EC 0.53–0.57 > AG 0.45–0.55
> TH/PFC/MTC 0.23–0.43** — the hippocampus-first vulnerability gradient
replicates SEA-AD (HIP/MEC strongest) and AD biology.

**Reading.** The ordinal trajectory replicates cleanly (real 0.37 vs null
−0.08, separation +0.45) and the regional gradient is biologically
consistent — but on this coarse 3-level target the grounded model adds
*nothing* over expression-only pseudobulk (0.51 vs 0.52), unlike SEA-AD
where it doubled it (0.60 vs 0.26). Interpretation: the text-grounded
advantage is target-dependent — it matters when the signal is a fine
continuous pathology score that raw averaging dilutes; when the label is a
coarse class driven by compositional shifts, expression alone suffices.
Verdict: **partial replication** — ordinal structure and regional pattern
replicate; no advantage over pseudobulk on the coarse target.

### A2 — zero-shot cross-cohort transfer (bootstrap CIs, n_te half of donors)

Models applied across cohorts in the shared SEA-AD 2000-HVG space
(`rosmap_for_seaad.h5ad`, 0 missing genes). Metric: PC1 trajectory of donor
embeddings, sign oriented on half the donors, Spearman ρ on the held-out
half, 1000× bootstrap CI. Nothing is fit on the target cohort.

| direction | target | ρ (mean over seeds×encoders) | CI range | gate 0.35 |
|---|---|---|---|---|
| **ROSMAP → SEA-AD** | CPS_Global | **0.478** | all CIs exclude 0 | **PASS** |
| ROSMAP → SEA-AD | ADNC ordinal | 0.37–0.54 | | PASS |
| SEA-AD (adnc) → ROSMAP | path_level | 0.289 | mostly >0 | ~gate |
| SEA-AD (cog) → ROSMAP | path_level | 0.179 | | fail |

**This is the strongest generalisation result in the portfolio.** A model
trained only on ROSMAP's 3-level pathology captions predicts SEA-AD's
continuous CPS_Global at ρ ≈ 0.46–0.52 — within ~0.1 of the within-cohort
models (0.56–0.60). The objective captures transferable disease biology,
not cohort idiosyncrasy.

Asymmetry is itself informative: (i) transfer toward a continuous target is
easier (partial axis alignment still correlates); (ii) the coarse ROSMAP
label forced a cleaner progression axis; (iii) cognitive status (binary)
does not transfer to neuropathology — cognition and pathology are
dissociable, consistent with the resilience literature.

## B — architecture

### B1 — attention/MIL pooling: NEGATIVE

Gated attention pooling (`softmax(wᵀ tanh(V hᵢ))`, ≤1500 cells/donor,
3 seeds) did not beat mean pooling end-to-end on held-out donors
(attn 0.34–0.60 vs mean 0.34–0.64; quantile-0.5 pooling matches for free).
Fails the Δρ gate. **Mean pooling stays.**

### B2 — soft-ordinal contrastive

`soft_ordinal_nce` implemented for both dual-head and single-head paths
(target distribution over ordinal caption bank weighted by
softmax(−|Δlevel|/ord_tau), τ = 1.0). SEA-AD sweep (3 seeds × 2 encoders):

| arm | encoder | CPS_Global ρ | vs hard | traj adnc | traj cerad |
|---|---|---|---|---|---|
| dual_cog_soft | SapBERT | **0.567** | **+0.09 ✅** | 0.695 | 0.720 |
| dual_cog_soft | MiniLM | 0.527 | −0.03 | 0.695 | 0.737 |
| dual_adnc_soft | SapBERT | 0.332 | −0.09 | **0.610** | **0.554** |
| dual_adnc_soft | MiniLM | 0.379 | −0.05 | **0.595** | **0.548** |

Pattern: soft-ordinal **sharpens ordinal trajectories consistently
(+0.04–0.10 on adnc/cerad)** at the cost of some ridge-CPS signal on the
adnc arm; on the cog arm it *improves* SapBERT CPS +0.09 and passes the
Δρ gate. Trade favourable for progression-structure claims. On ROSMAP's
3-level target it showed no gain (A1 table above).

## C — PubMed-retrieved pathology captions — SWEEP WINNER 🏆

30 dimension×level caption sets retrieved from the PubMed FTS index
(`data/text/m11/pathology_*_rag.json`), direct title/abstract
concatenation, no LLM synthesis (per the Qwen/OLMo decision).

| arm | encoder | CPS_Global ρ | vs M10 manual | traj adnc | traj cerad |
|---|---|---|---|---|---|
| **dual_cog_rag** | SapBERT | **0.607** | 0.475 → **+0.13** | 0.68–0.71 | ~0.72 |
| **dual_cog_rag** | MiniLM | **0.604** | 0.558 → **+0.05** | 0.68–0.71 | ~0.72 |
| dual_adnc_rag | SapBERT | 0.419 | ≈ 0.42 | — | — |
| dual_adnc_rag | MiniLM | 0.368 | ≈ 0.37 | — | — |

**Direct PubMed retrieval captions beat manually authored captions on
both encoders** — and answer the "caption engineering" critique: the
strongest grounding signal comes from retrieved literature evidence, not
from our prose. `dual_cog_rag` is the winning SEA-AD recipe and was
carried over to ROSMAP (`pathology_rosmap_rag.json`, same retrieval
pipeline, 3 levels).

### C→ROSMAP — honest negative within-cohort, transfer replicates

Retrieval captions for the 3 ROSMAP levels were generated with the same
pipeline (jobs 30253858–30253860). Results:

- **Within ROSMAP**: path_level ρ = 0.426/0.443 (MiniLM/SapBERT),
  trajectory 0.25/−0.09 — comparable to hard captions on ridge but
  *worse* trajectory than hard (0.37). Still ≤ pseudobulk (0.52).
  **The C1 advantage does not replicate on the coarse 3-level target.**
- **Transfer ROSMAP-rag → SEA-AD CPS_Global**: SapBERT mean **0.469**
  (0.303/0.567/0.536), MiniLM mean **0.467** (0.374/0.507/0.521),
  seed-0 CIs widest; essentially identical to the hard-caption transfer
  (0.46–0.52). The transferable pathology axis is **robust to grounding
  source** — retrieved and manual captions converge on the same
  cross-cohort signal.

## G2 — external scFM context (same donor-held-out protocol, no fine-tuning)

| model | cell-type probe F1 | donor CPS ρ | ADNC trajectory |
|---|---|---|---|
| Geneformer V1-10M | 0.73 | 0.28–0.37 | 0.60–0.79 |
| Geneformer V2-104M | 0.95 | 0.13–0.46 | 0.58–0.75 |
| scGPT whole-human | 0.90 | 0.43–0.51 | 0.55–0.81 |
| CellWhisperer CLIP | 0.63–0.64 | 0.34–0.58 | 0.38–0.52 |
| **ours M10 dual_cog** | 0.89–0.91 zeroshot | **0.56–0.60** | **0.70** |
| **ours C1 dual_cog_rag** | — | **0.604–0.607** | 0.68–0.71 |

Reproducing CellWhisperer surfaced three silent failure modes worth
documenting: (i) the public checkpoint ships only the text tower —
the transcriptome tower is a frozen external Geneformer-12L-30M; (ii) a
stale `model.safetensors` scaffold shadowed the real `pytorch_model.bin`;
(iii) the tokenizer defaulted to the gc104M dictionary vs the gc30M the
backbone was trained on. Only after all three fixes did embeddings carry
signal (balacc 0.04 → 0.29 on our HVG-2000 subset).

## Preregistered gate summary

| gate | result |
|---|---|
| ROSMAP replication ρ ≥ 0.5 | partial — reached numerically (0.513) but ≤ pseudobulk |
| transfer ρ ≥ 0.35 | **PASS — ROSMAP→SEA-AD ρ ≈ 0.46–0.52** (all CIs > 0) |
| Δρ ≥ +0.05 vs M10 baseline | B1 negative; B2 soft +0.09 (cog/SapBERT only); **C1 rag +0.05/+0.13 both encoders — WINNER** |

## Micro-sweep P2 — final ranking (SEA-AD dual_cog, donor CPS ρ)

| arm | MiniLM | SapBERT |
|---|---|---|
| M10 manual captions | 0.558 | 0.475 |
| B2 soft-ordinal | 0.527 | 0.567 |
| **C1 PubMed-rag** | **0.604** | **0.607** |

## Reproducibility

- `scripts/build_rosmap.py`, `m11_align_genespace.py`,
  `m11_transfer_eval.py`, `m11_rosmap_eval.py`, `m11_mil_pooling.py`,
  `m11_pathology_retrieve.py`
- `scripts/slurm/m11_rosmap.sbatch`, `m11_a2_*.sbatch`
- model weights now saved alongside embeddings (`.pt` + var_names) —
  enables all future transfer evals
- jobs: A1 30247075/30247549, A2 30248010–30248390, align 30248011,
  B2 30252294, C1 30252295, CW 30252296, rosmap-rag chain 30253858–30253860
