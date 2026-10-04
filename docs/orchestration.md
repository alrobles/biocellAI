# R7-ORCHESTRATION — SLURM DAG orchestration

The v2 pipeline runs as a declared DAG, not ad-hoc sbatch sequences.
`scripts/v2_pipeline.py` encodes the jobs and edges executed for the
confirmation pipeline and provides the three operations ADR-011 requires.

## Interface

```bash
python scripts/v2_pipeline.py --pipeline confirmation --dry-run
python scripts/v2_pipeline.py --pipeline confirmation --submit
python scripts/v2_pipeline.py --pipeline confirmation --validate
```

- `--dry-run` prints every `sbatch --parsable --dependency afterok:...`
  command and each job's declared deps without submitting — the plan is
  reviewable before compute is spent.
- `--submit` submits in declaration order (topological by construction —
  tests assert every dep is declared earlier) and wires `afterok` to the
  real job IDs returned by sbatch. A FAILED or never-satisfied dependency
  blocks its downstream subtree automatically.
- `--validate` checks each job's expected artifacts on the shared
  filesystem and reports `COMPLETED`/`INCOMPLETE` — a job counts as
  finished only when its declared outputs exist.

## Verified properties (acceptance criteria)

| Requirement | Evidence |
|---|---|
| Verifiable dry run | `--dry-run` emits the full 34-job DAG + afterok edges |
| Failure blocks downstream | afterok on every non-root job; observed when the first PFC run leg failed — dependent nulls/transfers never launched |
| Validation before COMPLETED | `--validate` checks expected artifacts; run on HPC → 34/34 COMPLETED (900 perm files, 9 run manifests, 9 transfer metrics, inference outputs) |
| GPU policy (`pilot 1, initial max 2`) | The v2 DAG is CPU-only by design (`--device cpu` in the run sbatch); no GPU requested anywhere in the confirmation chain |
| Logs on shared filesystem | every sbatch writes `--output/--error` to `/beegfs/a474r867/bioai/logs/`; no remote /tmp polling |
| Atomic/idempotent outputs | `reserve_output` refuses existing dirs; the perm-null writer refuses to overwrite `perm_<k>.csv` inside a shared array dir — retried transient node failures produced no duplicates |
| No secrets in logs | grep over the v2* log family finds no key/token strings |

## Executed run (reference)

The confirmation chain was submitted with the same edges the DAG encodes:
prep `30798624` → captions `30798625` → runs `30798627-29` → nulls
`30799068-70` + `30834427-33` → transfers `30799071-73` + `30834467-68` →
inference `30834472`. MTG/ROSMAP legs used the earlier `30792xxx` chain.

Retried pieces were resubmitted with explicit `--array=i,j,k` lists for
the missing indices only; the per-file guard made that safe.
