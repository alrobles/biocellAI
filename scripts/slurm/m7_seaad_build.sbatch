#!/bin/bash
#SBATCH --job-name=m7_build
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m7_build_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m7_build_%j.err
#SBATCH --partition=sixhour
#SBATCH --time=4:00:00
#SBATCH --mem=224G
#SBATCH --cpus-per-task=8

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

python scripts/build_seaad.py \
    --data-dir "$ROOT/data" \
    --max-cells 240000 \
    --regions MTG V1C \
    --seed 0

echo "m7 build done"
