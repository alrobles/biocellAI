#!/bin/bash
#SBATCH --job-name=m7_prog
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m7_prog_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m7_prog_%j.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

EMB=${EMB:-experiments/m7_rag_markers_alias/embeddings}
TAG=${TAG:-$(echo "$EMB" | tr '/' '_')}

python scripts/m7_progression.py \
    --emb-dir "$EMB" \
    --donor-table "$ROOT/data/seaad_donors.csv" \
    --h5ad "$ROOT/data/seaad_mtg_v1c_240026_s0.h5ad" \
    --out "experiments/m7_progression_${TAG}"

echo "progression $TAG done"
