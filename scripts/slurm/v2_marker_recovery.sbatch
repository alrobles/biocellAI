#!/bin/bash
#SBATCH --job-name=v2markrec
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2markrec_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2markrec_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# R5-MARKERS: non-circular marker recovery. One job = one cohort across
# all its folds (revalidation + confirmation trees).
#   sbatch --export=ALL,COHORT=seaad_mtg v2_marker_recovery.sbatch
#   COHORT in {blood, seaad_mtg, rosmap, seaad_pfc}

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

COHORT=${COHORT:?required}
OUT=experiments/v2_marker_recovery
REV=experiments/v2_revalidation
CONF=experiments/v2_confirmation

case "$COHORT" in
  blood)     REF=data/manifests/panglao_markers.json ;;
  rosmap)    REF=data/manifests/panglao_rosmap_markers.json ;;
  seaad_*)   REF=data/manifests/panglao_brain_markers.json ;;
  *) echo "unknown COHORT=$COHORT"; exit 2 ;;
esac

FOLD_ARGS=(); CAP_ARGS=()
case "$COHORT" in
  seaad_mtg|rosmap|blood)
    for s in 0 1 2; do
      FOLD_ARGS+=(--fold "$REV/$COHORT/folds/fold_s${s}.h5ad")
      CAP_ARGS+=(--captions "$REV/$COHORT/folds/captions_s${s}_markers.json")
    done
    ;;
esac
case "$COHORT" in
  seaad_mtg|rosmap) SEEDS="3 4 5" ;;
  seaad_pfc)        SEEDS="0 1 2" ;;
  *)                SEEDS="" ;;
esac
for s in $SEEDS; do
  FOLD_ARGS+=(--fold "$CONF/$COHORT/folds/fold_s${s}.h5ad")
  CAP_ARGS+=(--captions "$CONF/$COHORT/folds/captions_s${s}_markers.json")
done

python scripts/v2_marker_recovery.py \
    "${FOLD_ARGS[@]}" "${CAP_ARGS[@]}" \
    --reference "$REF" \
    --cohort "$COHORT" \
    --out "$OUT"

echo "v2 marker recovery $COHORT done -> $OUT"
