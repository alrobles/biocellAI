#!/bin/bash
#SBATCH --job-name=v2conf_xfer_pfc
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2conf_xfer_pfc_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2conf_xfer_pfc_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# R6-CONFIRMATION extra transfer leg: frozen MTG-fitted artifacts applied
# strictly to never-analyzed SEA-AD PFC cells (donor overlap with MTG is
# documented in spec/confirmation_plan.md).
#   sbatch --export=ALL,SEED=0 v2_confirm_transfer_pfc.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1

SEED=${SEED:?required}
EXP=experiments/v2_revalidation
CONF=experiments/v2_confirmation

python scripts/v2_transfer.py \
    --source-fold "$EXP/seaad_mtg/folds/fold_s${SEED}.h5ad" \
    --source-run "$EXP/seaad_mtg/run_s${SEED}_full" \
    --target-raw "$ROOT/data/v2_src/seaad_pfc.h5ad" \
    --target-fold "$CONF/seaad_pfc/folds/fold_s${SEED}.h5ad" \
    --source-cohort seaad_mtg \
    --target-cohort seaad_pfc \
    --target-pathology CPS_Global \
    --arms "B0,B1,B2,T0,T1,T2,T3,N1,N2" \
    --seed "$SEED" \
    --device cpu \
    --out "$CONF/transfer/seaad_mtg_to_seaad_pfc/s${SEED}"

echo "v2 confirm transfer seaad_mtg -> seaad_pfc s$SEED done"
