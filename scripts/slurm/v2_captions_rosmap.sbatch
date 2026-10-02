#!/bin/bash
#SBATCH --job-name=v2caps_rosmap
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2caps_rosmap_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2caps_rosmap_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# V2 R1-CAPTIONS: fold-specific caption banks for ROSMAP folds.
#   markers:    train-donor-only DE markers (v2_captions --mode markers)
#   pubmed_rag: SEA-AD allregions retrieval corpus under the documented
#               ROSMAP->SEA-AD Subclass name map (32/34 covered; unmapped
#               classes degrade to name-only via --missing-policy name)
# Run with:  sbatch --dependency=afterok:<prep_jid> v2_captions_rosmap.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

FOLDS=experiments/v2_revalidation/rosmap/folds
CORPUS=data/text/m7/type_desc_pubmed_rag_markers_alias_rosmap.json

python scripts/v2_captions.py \
    --fold "$FOLDS/fold_s0.h5ad" \
    --fold "$FOLDS/fold_s1.h5ad" \
    --fold "$FOLDS/fold_s2.h5ad" \
    --mode markers --label-col cell_type

python scripts/v2_captions.py \
    --fold "$FOLDS/fold_s0.h5ad" \
    --fold "$FOLDS/fold_s1.h5ad" \
    --fold "$FOLDS/fold_s2.h5ad" \
    --mode copy --source "$CORPUS" \
    --missing-policy name --tag pubmed_rag

echo "v2 ROSMAP captions done"
