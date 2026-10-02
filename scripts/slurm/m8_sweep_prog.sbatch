#!/bin/bash
#SBATCH --job-name=m8_sprog
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m8_sprog_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m8_sprog_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6

# M8 sweep progression eval — same array indexing as m8_sweep.sbatch.

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

TAGS=(braak cerad thal cog cps_q3 cps_q4 combo)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

python scripts/m7_progression.py \
    --emb-dir "experiments/m8_sweep_${TAG}/embeddings" \
    --donor-table "$ROOT/data/seaad_donors.csv" \
    --h5ad "$ROOT/data/seaad_mtg_v1c_240026_s0.h5ad" \
    --out "experiments/m8_sweep_${TAG}/progression"

echo "progression m8_sweep_${TAG} done"
