#!/bin/bash
#SBATCH --job-name=v2conf_run
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2conf_run_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2conf_run_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# R6-CONFIRMATION arm-matrix run on a confirmation fold (frozen v2 config).
# Mirrors v2_run.sbatch but under experiments/v2_confirmation/.
# N2 permutes the cohort's original arm: rosmap -> T2, seaad_* -> T3.
#   sbatch --export=ALL,COHORT=seaad_mtg,SEED=3 v2_confirm_run.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1

COHORT=${COHORT:?required}
SEED=${SEED:?required}
OUTBASE=experiments/v2_confirmation
FOLDDIR=$OUTBASE/$COHORT/folds
FOLD=$FOLDDIR/fold_s${SEED}.h5ad
OUT=$OUTBASE/$COHORT/run_s${SEED}_full

case "$COHORT" in
  seaad_mtg|seaad_pfc) PATH_COL=CPS_Global; N2_OF=T3 ;;
  rosmap)              PATH_COL=path_level; N2_OF=T2 ;;
  *) echo "unknown COHORT=$COHORT"; exit 2 ;;
esac

python scripts/v2_run.py \
    --fold "$FOLD" --out "$OUT" \
    --arms "B0,B1,B2,T0,T1,T2,T3,N1,N2" \
    --label-col cell_type --pathology-col "$PATH_COL" \
    --captions "T2=$FOLDDIR/captions_s${SEED}_markers.json" \
    --captions "T3=$FOLDDIR/captions_s${SEED}_pubmed_rag.json" \
    --n2-of "$N2_OF" \
    --text-encoder all-MiniLM-L6-v2 \
    --epochs 40 --device cpu --seed "$SEED"

echo "v2 confirm run $COHORT s$SEED done -> $OUT"
