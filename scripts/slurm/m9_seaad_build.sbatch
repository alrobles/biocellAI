#!/bin/bash
#SBATCH --job-name=m9_build
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m9_build_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m9_build_%j.err
#SBATCH --partition=sixhour
#SBATCH --time=6:00:00
#SBATCH --mem=224G
#SBATCH --cpus-per-task=8

# Gen-2: full-region SEA-AD working set — 11 regions x ~40k cells.
# Same loader/preprocess as M7; separate donor table + output names so
# the MTG+V1C artifacts stay untouched.

set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/beegfs/a474r867/biocellai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

python scripts/build_seaad.py \
    --data-dir "$ROOT/data" \
    --max-cells 440000 \
    --regions MTG V1C AnG HIP ITG MEC LEC STG FI PFC \
    --seed 0 \
    --out "$ROOT/data/seaad_allregions_s0.h5ad" \
    --donor-out "$ROOT/data/seaad_donors_allregions.csv" \
    --manifest-out "$ROOT/data/seaad_manifest_allregions.json" \
    --markers-out data/manifests/markers_seaad_allregions_s0.json

echo "m9 build done"
