#!/bin/bash
#SBATCH --job-name=v2prep_seaad
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2prep_seaad_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2prep_seaad_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=5:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=8

# V2 R1-REAL-DATA: SEA-AD MTG
#   1) extract verified UMI counts (layers/UMIs) with normalized obs schema
#      (Donor ID -> donor_id, Subclass -> cell_type, donor table join adds
#      CPS_Global etc.), capped at 1500 cells/donor
#   2) build donor-held-out folds with train-only QC/HVG via v2_prepare

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

SRC=$ROOT/data/SEAAD_MTG_RNAseq_final-nuclei.2026-06-22.h5ad
EXTRACT=$ROOT/data/v2_src/seaad_mtg
FOLDS=$ROOT/repos/biocellai-devel/experiments/v2_revalidation/seaad_mtg/folds

if [ ! -f "${EXTRACT}.h5ad" ]; then
    python scripts/v2_extract.py \
        --input "$SRC" \
        --counts-layer UMIs \
        --donor-col "Donor ID" \
        --rename-obs "Subclass=cell_type" \
        --keep-obs "Donor ID,Subclass,Class,Cognitive Status,Braak,CERAD score,Brain Region" \
        --donor-table "$ROOT/data/seaad_donors.csv" \
        --donor-table-key donor_id \
        --max-cells-per-donor 1500 \
        --seed 0 \
        --out "$EXTRACT"
fi

python scripts/v2_prepare.py \
    --input "${EXTRACT}.h5ad" \
    --counts-layer X \
    --cohort seaad \
    --source-release "SEA-AD MTG final-nuclei 2026-06-22 (UMIs layer)" \
    --seeds 0 1 2 \
    --n-hvg 2000 \
    --min-genes 200 \
    --min-cells 3 \
    --out "$FOLDS"

echo "v2 SEA-AD MTG prep done"
