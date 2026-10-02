# Historical results — honest review for credibility

All results below are **exploratory / pre-v2**. They were produced under the
historical protocol (preprocessed inputs, partially transductive transforms,
seed-0-derived captions reused across splits) and are preserved as evidence
of direction, not as validated claims. Full context: [milestones.md](milestones.md).

## What the record shows

### 1. Blood cell identity — text-mediated supervision works

| Arm | macro-F1 | Notes |
|---|---|---|
| Supervised classifier (M3, 85k cells) | 0.603 | Direct label baseline |
| Marker captions → zero-shot class proto (MiniLM) | **0.591** | Closest honest arm; markers derived from *train* labels |
| RAG → LLM prose, name stripped (`ragllm_honest`) | 0.589 | External knowledge, no class name in text |
| PanglaoDB → retrieval → LLM (`ragllm_curated`) | **0.603** | Matches supervised ceiling on this task |
| Per-cell top genes as caption | 0.29 | Honest but weak |
| Label-name-in-text probes | 0.51–0.60 | Leakage controls, not claims |

**Honest reading:** captions can carry enough signal to reach supervised-level
cell-ID, but every caption was *assigned via the label*. This is
text-mediated supervision — it does not yet show biology adds anything beyond
the class identity itself. The v2 one-hot/random-prototype arms exist to test
exactly that.

### 2. SEA-AD donor pathology — a promising, unvalidated axis

| Result | Value | Status |
|---|---|---|
| C1 dual-head, CPS correlation (held-out donors) | ρ ≈ 0.604–0.607 | Best historical arm |
| Ordinal trajectory (CERAD) | ρ up to ~0.72, positive across seeds | Directionally consistent |
| Regional vulnerability pattern | HC > EC > cortex gradient | Reproduces known biology coarsely |
| Shuffle-null pooled embeddings | ρ 0.2–0.38 retained | Nulls are strong — donor pooling carries signal |
| Cell-level pathology classification | ~chance (0.23 F1) | Signal is at the **donor**, not cell, level |

### 3. ROSMAP replication — the sobering control

| Arm | path-level ρ |
|---|---|
| Hard InfoNCE (real) | 0.513 |
| **Pseudobulk PCA (expression only)** | **0.517** |
| Shuffled null | 0.498 |

The grounded model adds **nothing over pseudobulk** on ROSMAP, and the null
nearly matches the real arm. Regional gradient (HC > EC > cortex) did
replicate biologically — the single most credibility-relevant finding.

### 4. scGPT scaling (M12) — a clean negative result

| Arm | CPS ρ |
|---|---|
| S1 frozen scGPT + dual-head | 0.586 |
| S2 full fine-tune | 0.543 |
| S2 last-2 PEFT | 0.528 |
| C1 MLP from scratch | **0.604** |

No tested scGPT configuration beat the small MLP; raw frozen scGPT features
retain ρ ≈ 0.47–0.54 under a *null*, so features carry donor structure
largely independent of the objective. **Bounded claim:** "these scGPT
configurations did not help," not "scale doesn't matter."

### 5. Composition baseline (M13 diagnostic)

Cell-type fractions alone: **ρ ≈ 0.504**. The apparent pathology axis is
largely compositional; marginal contribution of within-type state over
composition is small and requires the adjusted v2 analysis to quantify.

## Credibility assets (why this record helps rather than hurts)

- **Published negative results**: M12 underperforming, ROSMAP pseudobulk
  parity, cell-level pathology at chance — kept and reported, not buried.
- **Leakage probes disclosed**: label-in-text arms measured and labeled as
  upper bounds rather than hidden.
- **Self-audit**: the project caught its own preprocessing/split/caption
  leakage and rebuilt the protocol (ADR-011) instead of shipping optimistic
  numbers — exactly the rigor a reviewer for a rigor-focused position wants.
- **Biologically coherent pattern replicated across cohorts** (hippocampal
  vulnerability gradient), the strongest qualitative signal.

## Credibility risks (must keep visible)

- "0.59 ≈ supervised 0.60" is text-**mediated** supervision, not label-free
  grounding.
- Strong shuffle nulls (0.38–0.54) mean some donor-level signal is trivially
  pooled structure; headline numbers need paired-null comparison.
- ρ differences were not computed on identical donor sets in all arms and
  some gates were called on rounded figures — all corrected in v2 design.
- No claim should survive contact with a reviewer unless labeled historical.
