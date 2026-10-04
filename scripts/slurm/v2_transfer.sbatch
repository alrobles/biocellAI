#!/bin/bash
#SBATCH --job-name=v2xfer
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2xfer_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2xfer_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# V2 R2-TRANSFER: strict source-fitted transfer + calibrated refit.
# Parametrized at submission:
#   sbatch --export=ALL,SRC=rosmap,TGT=seaad_mtg,SEED=0 v2_transfer.sbatch
# STRICT: encoder/normalization/axis/sign all source-fitted; the target
# supplies cells only. REFIT re-orients the donor readout on target train
# donors (supervised orientation — separate analysis per ADR-011).

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1

SRC=${SRC:?required}
TGT=${TGT:?required}
SEED=${SEED:-0}
OUTBASE=${OUTBASE:-experiments/v2_revalidation}

case "$TGT" in
  seaad_mtg) TRAW=$ROOT/data/v2_src/seaad_mtg.h5ad;       PATH_COL=CPS_Global ;;
  seaad_pfc) TRAW=$ROOT/data/v2_src/seaad_pfc.h5ad;       PATH_COL=CPS_Global ;;
  rosmap)    TRAW=$ROOT/data/v2_src/rosmap_liu2025.h5ad;  PATH_COL=path_level ;;
  *) echo "unknown TGT=$TGT"; exit 2 ;;
esac

SFOLD=$OUTBASE/$SRC/folds/fold_s${SEED}.h5ad
SRUN=$OUTBASE/$SRC/run_s${SEED}_full
TFOLD=$OUTBASE/$TGT/folds/fold_s${SEED}.h5ad
OUT=$OUTBASE/transfer/${SRC}_to_${TGT}/s${SEED}

python scripts/v2_transfer.py \
    --source-fold "$SFOLD" \
    --source-run "$SRUN" \
    --target-raw "$TRAW" \
    --target-fold "$TFOLD" \
    --source-cohort "$SRC" \
    --target-cohort "$TGT" \
    --target-pathology "$PATH_COL" \
    --arms "B0,B1,B2,T0,T1,T2,T3,N1,N2" \
    --seed "$SEED" \
    --device cpu \
    --out "$OUT"

echo "v2 transfer $SRC -> $TGT seed $SEED done -> $OUT"
