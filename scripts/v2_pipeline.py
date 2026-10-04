#!/usr/bin/env python3
"""R7-ORCHESTRATION: declarative SLURM DAG driver for the v2 pipeline.

Encodes the confirmation DAG (prep -> captions -> runs -> nulls ->
transfers -> inference) as job records so the dependency graph is
reviewable instead of reconstructed from ad-hoc submission history.

  --dry-run   print every sbatch command + afterok edges; submits nothing
  --submit    run the DAG: each job's afterok wires to its declared deps,
              so a FAILED/never-satisfied dependency blocks downstream work
  --validate  check expected artifacts on the shared filesystem; a job is
              COMPLETED only if its expected outputs exist

All jobs log to /beegfs shared logs (per-sbatch directives); array jobs
stay within the GPU policy (no GPU in this DAG; arrays are CPU).

Usage:
    python scripts/v2_pipeline.py --pipeline confirmation --dry-run
    python scripts/v2_pipeline.py --pipeline confirmation --submit
    python scripts/v2_pipeline.py --pipeline confirmation --validate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLURM = Path("scripts/slurm")  # relative: sbatch runs from the repo dir
CONF = Path("experiments/v2_confirmation")


@dataclass
class Job:
    jid: str                      # internal id used for deps
    sbatch: str                   # script name under scripts/slurm
    env: dict = field(default_factory=dict)
    array: str | None = None
    deps: list = field(default_factory=list)
    expect: list = field(default_factory=list)  # artifacts for --validate

    def cmd(self, dep_ids: dict) -> list:
        c = ["sbatch", "--parsable"]
        if self.array:
            c += ["--array", self.array]
        if self.deps:
            ok = ":".join(dep_ids[d] for d in self.deps)
            c += ["--dependency", f"afterok:{ok}"]
        env = ",".join(f"{k}={v}" for k, v in self.env.items())
        c += ["--export", f"ALL,{env}", str(SLURM / self.sbatch)]
        return c


def confirmation_dag() -> list[Job]:
    """The exact DAG executed for R6-CONFIRMATION."""
    jobs: list[Job] = []
    runs, nulls, xfers = {}, {}, {}

    def per_cohort(cohort, path_col, seeds):
        jobs.append(Job(
            f"prep_{cohort}", "v2_confirm_prep.sbatch",
            env={"COHORT": cohort},
            expect=[f"{CONF}/{cohort}/folds/fold_s{s}.json" for s in seeds]))
        jobs.append(Job(
            f"caps_{cohort}", "v2_confirm_captions.sbatch",
            env={"COHORT": cohort, "SEEDS": " ".join(map(str, seeds))},
            deps=[f"prep_{cohort}"],
            expect=[f"{CONF}/{cohort}/folds/captions_s{s}_markers.json"
                    for s in seeds]))
        for s in seeds:
            rid = f"run_{cohort}_s{s}"
            runs[(cohort, s)] = rid
            jobs.append(Job(
                rid, "v2_confirm_run.sbatch",
                env={"COHORT": cohort, "SEED": s},
                deps=[f"caps_{cohort}"],
                expect=[f"{CONF}/{cohort}/run_s{s}_full/metrics.csv",
                        f"{CONF}/{cohort}/run_s{s}_full/manifest.json"]))
            nid = f"null_{cohort}_s{s}"
            nulls[(cohort, s)] = nid
            jobs.append(Job(
                nid, "v2_permnull.sbatch", array="0-99",
                env={"COHORT": cohort, "SEED": s, "PATH_COL": path_col,
                     "OUTBASE": str(CONF)},
                deps=[rid],
                expect=[f"{CONF}/{cohort}/perm_null_s{s}/perm_{i}.csv"
                        for i in range(100)]))

    per_cohort("seaad_mtg", "CPS_Global", (3, 4, 5))
    per_cohort("rosmap", "path_level", (3, 4, 5))
    per_cohort("seaad_pfc", "CPS_Global", (0, 1, 2))

    # strict transfers: confirmation sources + frozen exploratory MTG->PFC
    for src, tgt, seeds in (("rosmap", "seaad_mtg", (3, 4, 5)),
                            ("seaad_mtg", "rosmap", (3, 4, 5))):
        for s in seeds:
            xid = f"xfer_{src}_to_{tgt}_s{s}"
            xfers[(src, tgt, s)] = xid
            jobs.append(Job(
                xid, "v2_transfer.sbatch",
                env={"SRC": src, "TGT": tgt, "SEED": s,
                     "OUTBASE": str(CONF)},
                deps=[runs[(src, s)], f"prep_{tgt}"],
                expect=[f"{CONF}/transfer/{src}_to_{tgt}/s{s}/metrics.csv"]))
    for s in (0, 1, 2):
        xid = f"xfer_seaad_mtg_to_seaad_pfc_s{s}"
        xfers[("seaad_mtg", "seaad_pfc", s)] = xid
        jobs.append(Job(
            xid, "v2_confirm_transfer_pfc.sbatch",
            env={"SEED": s}, deps=[f"prep_seaad_pfc"],
            expect=[f"{CONF}/transfer/seaad_mtg_to_seaad_pfc/s{s}/metrics.csv"]))

    jobs.append(Job(
        "infer", "v2_confirm_infer.sbatch",
        deps=sorted(set(nulls.values()) | set(xfers.values())),
        expect=[f"{CONF}/inference/gates.json",
                f"{CONF}/inference/paired_comparisons.csv",
                f"{CONF}/inference/manifest.json"]))
    return jobs


def dry_run(jobs: list[Job]) -> None:
    print(f"# {len(jobs)} jobs (submit order = declaration order)\n")
    fake = {j.jid: f"<{j.jid}>" for j in jobs}
    for j in jobs:
        print(f"[{j.jid}] deps={j.deps or 'none'}")
        print("  " + " ".join(j.cmd(fake)) + "\n")


def submit(jobs: list[Job]) -> None:
    ids: dict[str, str] = {}
    for j in jobs:
        out = subprocess.run(j.cmd(ids), capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(f"sbatch failed for {j.jid}: {out.stderr.strip()}")
        ids[j.jid] = out.stdout.strip()
        print(f"{j.jid:42s} -> {ids[j.jid]} (afterok: "
              f"{[ids[d] for d in j.deps] or 'none'})")
    (Path(".pipeline_job_ids.json")).write_text(json.dumps(ids, indent=1))


def validate(jobs: list[Job]) -> int:
    bad = 0
    for j in jobs:
        missing = [e for e in j.expect if not (ROOT / e).exists()]
        status = "COMPLETED" if not missing else "INCOMPLETE"
        if missing:
            bad += 1
        print(f"{j.jid:42s} {status:11s} "
              f"({len(j.expect) - len(missing)}/{len(j.expect)} artifacts)")
        for m in missing[:3]:
            print(f"{'':44s} missing: {m}")
    print(f"\n{bad} of {len(jobs)} jobs incomplete")
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["confirmation"], required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--validate", action="store_true")
    args = p.parse_args()

    jobs = confirmation_dag()
    if args.dry_run:
        dry_run(jobs)
    elif args.submit:
        submit(jobs)
    else:
        sys.exit(validate(jobs))


if __name__ == "__main__":
    main()
