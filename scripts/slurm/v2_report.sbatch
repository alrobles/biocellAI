#!/bin/bash
#SBATCH --job-name=v2report
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2report_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2report_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# V2 R6-INFERENCE: paired donor-level rho comparisons + improvement gates
# over the three seed runs of one cohort. Parametrized:
#   sbatch --dependency=afterok:j1:j2:j3 --export=ALL,COHORT=rosmap,TAG=full \
#          v2_report.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

COHORT=${COHORT:-seaad_mtg}
TAG=${TAG:-full}
BASE=experiments/v2_revalidation/${COHORT}

python scripts/v2_report.py \
    --runs "${BASE}/run_s0_${TAG}" "${BASE}/run_s1_${TAG}" "${BASE}/run_s2_${TAG}" \
    --pairs T2:B2 T3:B2 T2:T0 T2:T1 T2:N2 T0:B2 \
    --out "${BASE}/report"

echo "v2 report ${COHORT} done -> ${BASE}/report"
