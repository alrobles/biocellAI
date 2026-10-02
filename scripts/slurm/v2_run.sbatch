#!/bin/bash
#SBATCH --job-name=v2run
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2run_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2run_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# V2 arm-matrix runner for one prepared fold (ADR-011 R3-SMOKE / R3-MATRIX).
# Parametrized at submission:
#   sbatch --export=ALL,COHORT=seaad_mtg,SEED=0,ARMS=B0,B1,B2,T0,T1,T2,T3,N1,N2,\
#          PATH_COL=CPS_Global,CAPTIONS="T2=captions_s0_markers.json T3=captions_s0_pubmed.json",\
#          N2_OF=T2 v2_run.sbatch
# Captions paths are resolved relative to the fold directory.

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1

# NOTE: sbatch --export splits on commas — pass ARMS space-separated:
#   ARMS="B0 B1 B2 T0 T1 N1"
COHORT=${COHORT:-blood}
SEED=${SEED:-0}
ARMS=$(echo "${ARMS:-B0 B1 B2 T0 T1 N1}" | tr ' ' ',')
PATH_COL=${PATH_COL:-}
EPOCHS=${EPOCHS:-40}
DEVICE=${DEVICE:-cpu}
OUTBASE=${OUTBASE:-experiments/v2_revalidation}

FOLDDIR=experiments/v2_revalidation/${COHORT}/folds
FOLD=${FOLDDIR}/fold_s${SEED}.h5ad
TAG=${TAG:-$(echo "${ARMS}" | tr ',A-Z' ' a-z' | tr ' ' '-')}
OUT=${OUTBASE}/${COHORT}/run_s${SEED}_${TAG}

CAP_ARGS=()
for spec in ${CAPTIONS:-}; do
    arm="${spec%%=*}"; file="${spec#*=}"
    CAP_ARGS+=(--captions "${arm}=${FOLDDIR}/${file}")
done

EXTRA=()
if [ -n "$PATH_COL" ]; then EXTRA+=(--pathology-col "$PATH_COL"); fi
if [ -n "${N2_OF:-}" ]; then EXTRA+=(--n2-of "$N2_OF"); fi

python scripts/v2_run.py \
    --fold "$FOLD" \
    --out "$OUT" \
    --arms "$ARMS" \
    --label-col "${LABEL_COL:-cell_type}" \
    "${CAP_ARGS[@]}" "${EXTRA[@]}" \
    --text-encoder "${TEXT_ENCODER:-all-MiniLM-L6-v2}" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --seed "$SEED"

echo "v2 run ${COHORT} s${SEED} arms=${ARMS} done -> ${OUT}"
