#!/bin/bash
#SBATCH --job-name=v2conf_caps
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2conf_caps_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2conf_caps_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# R6-CONFIRMATION captions for the confirmation folds (same frozen modes):
#   markers     — train-donor-only DE captions (T2)
#   pubmed_rag  — copy of the registered corpus (T3):
#                 seaad_*  -> type_desc_pubmed_rag_markers_alias_allregions
#                 rosmap   -> type_desc_pubmed_rag_markers_alias_rosmap
#   sbatch --export=ALL,COHORT=seaad_mtg,SEEDS="3 4 5" v2_confirm_captions.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

COHORT=${COHORT:?required}
SEEDS=${SEEDS:-"3 4 5"}
OUTBASE=experiments/v2_confirmation
FOLDS=$OUTBASE/$COHORT/folds

case "$COHORT" in
  seaad*) CORPUS=data/text/m7/type_desc_pubmed_rag_markers_alias_allregions.json ;;
  rosmap) CORPUS=data/text/m7/type_desc_pubmed_rag_markers_alias_rosmap.json ;;
  *) echo "unknown COHORT=$COHORT"; exit 2 ;;
esac

FOLD_ARGS=()
for s in $SEEDS; do FOLD_ARGS+=(--fold "$FOLDS/fold_s${s}.h5ad"); done

python scripts/v2_captions.py "${FOLD_ARGS[@]}" \
    --mode markers --label-col cell_type

python scripts/v2_captions.py "${FOLD_ARGS[@]}" \
    --mode copy --source "$CORPUS" \
    --missing-policy name --tag pubmed_rag

echo "v2 confirm captions $COHORT seeds $SEEDS done"
