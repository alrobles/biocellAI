# R4-C2S scope decision

**Decision:** C2S-Scale is excluded from the executed R4 benchmark; the
audit documents the exclusion explicitly rather than silently omitting
it. This does not remove or weaken the G2 requirement (native external
scFM comparison): scGPT, Geneformer, and CellWhisperer are all executed
on the corrected v2 folds.

## Why C2S cannot be benchmarked reproducibly here

1. **No weights present.** C2S-Scale (google/Cell2Sentence-Scale, the
   Gemma-family scFM) has no checkpoint under
   `/beegfs/a474r867/bioai/hf_cache` and none was recorded in
   `experiments/v2_revalidation/inventory.json` (R0-INVENTORY, 3,744
   files catalogued).
2. **No offline acquisition path.** The v2 compute environment runs
   with `HF_HUB_OFFLINE=1` (provenance discipline); fetching a
   multi-GB proprietary-format checkpoint mid-benchmark would break
   environment immutability and cannot be hash-pinned against a
   pre-audited inventory entry.
3. **Size/class mismatch.** C2S models are 27B-class causal LLMs whose
   cell input is a text serialization, not a ranked token vocabulary —
   the "same cells, same probe" contract holds, but compute cost and
   checkpoint provenance would need a separate acquisition + audit
   cycle, not an add-on to this one.

## What satisfies the criterion instead

- `R4-BENCHMARK` compares native vs HVG inputs for the three audited
  external families under the identical probe/donor-ridge protocol —
  the G2 question ("does a *trained-in* language-relevant prior beat
  our arms?") is answered by Geneformer/scGPT/CellWhisperer already.
- If C2S weights are later acquired under the same inventory+hash
  discipline, `biocellai.scfm` accepts a new registry entry without
  protocol changes.

## Status

Scope decision — recorded 2026-09 (R4 audit). G2 stands: three
external families × {native, hvg} inputs × 3 cohorts × 3 seeds.
