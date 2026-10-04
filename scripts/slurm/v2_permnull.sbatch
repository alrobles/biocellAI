#!/bin/bash
#SBATCH --job-name=v2permnull
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2permnull_%x_%A_%a.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2permnull_%x_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# V2 R6-INFERENCE permutation nulls (ADR-011): 100 shared donor->label
# maps for the MLP matrix. One array task = one permutation index applied
# to every arm. B2 retrains on permuted train labels (N1 replicate); the
# other arms permute the donor-ridge target on frozen features.
#   sbatch --export=ALL,COHORT=seaad_mtg,SEED=0,PATH_COL=CPS_Global \
#          --array=0-99 v2_permnull.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1

COHORT=${COHORT:?required}
SEED=${SEED:?required}
PATH_COL=${PATH_COL:?required}
ARMS=$(echo "${ARMS:-B0 B1 B2 T0 T1 T2 T3 N2}" | tr ' ' ',')
OUTBASE=${OUTBASE:-experiments/v2_revalidation}
PERM=${SLURM_ARRAY_TASK_ID:?run as a SLURM array}

FOLD=$OUTBASE/$COHORT/folds/fold_s${SEED}.h5ad
RUN=$OUTBASE/$COHORT/run_s${SEED}_full
OUT=$OUTBASE/$COHORT/perm_null_s${SEED}

python scripts/v2_perm_null.py \
    --fold "$FOLD" \
    --run "$RUN" \
    --out "$OUT" \
    --perm-index "$PERM" \
    --arms "$ARMS" \
    --label-col "${LABEL_COL:-cell_type}" \
    --pathology-col "$PATH_COL" \
    --device cpu

echo "perm null ${COHORT} s${SEED} perm=${PERM} done -> $OUT"
