#!/bin/bash
#SBATCH --job-name=m9_prog
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m9_prog_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m9_prog_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

TAGS=(cog adnc cerad braak)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

python scripts/m7_progression.py \
    --emb-dir "experiments/m9_${TAG}/embeddings" \
    --donor-table "$ROOT/data/seaad_donors_allregions.csv" \
    --h5ad "$ROOT/data/seaad_allregions_s0.h5ad" \
    --out "experiments/m9_${TAG}/progression"

echo "progression m9_${TAG} done"
