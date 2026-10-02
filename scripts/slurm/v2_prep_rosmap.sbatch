#!/bin/bash
#SBATCH --job-name=v2prep_rosmap
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2prep_rosmap_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2prep_rosmap_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# V2 R1-REAL-DATA: ROSMAP Liu2025 multiregion
#   1) extract verified X counts; ROSMAP_IndividualID -> donor_id,
#      RNA.Subclass -> cell_type; donor table join adds path_level etc.;
#      capped at 1500 cells/donor
#   2) donor-held-out folds via v2_prepare

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

SRC=$ROOT/data/rosmap_liu2025/snRNA_multiregion_liu2025.h5ad
EXTRACT=$ROOT/data/v2_src/rosmap_liu2025
FOLDS=$ROOT/repos/biocellai-devel/experiments/v2_revalidation/rosmap/folds

if [ ! -f "${EXTRACT}.h5ad" ]; then
    python scripts/v2_extract.py \
        --input "$SRC" \
        --counts-layer X \
        --donor-col "ROSMAP_IndividualID" \
        --rename-obs "RNA.Subclass=cell_type" \
        --keep-obs "ROSMAP_IndividualID,RNA.Class,RNA.Subclass,Pathology,BrainRegion,Sex,age_death,pmi" \
        --donor-table "$ROOT/data/rosmap_donors.csv" \
        --donor-table-key donor_id \
        --max-cells-per-donor 1500 \
        --seed 0 \
        --out "$EXTRACT"
fi

python scripts/v2_prepare.py \
    --input "${EXTRACT}.h5ad" \
    --counts-layer X \
    --cohort rosmap \
    --source-release "ROSMAP snRNA multiregion Liu2025 (X counts)" \
    --seeds 0 1 2 \
    --n-hvg 2000 \
    --min-genes 200 \
    --min-cells 3 \
    --out "$FOLDS"

echo "v2 ROSMAP prep done"
