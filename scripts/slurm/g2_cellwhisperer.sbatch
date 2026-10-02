#!/bin/bash
#SBATCH --job-name=g2_cellwhisperer
#SBATCH --output=/beegfs/a474r867/biocellai/logs/g2_cw_%A.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/g2_cw_%A.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:l40:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# G2: CellWhisperer CLIP-v1 zero-shot embeddings on the identical SEA-AD
# donor split. Env compat shims live inside scripts/g2_cellwhisperer.py.

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

python scripts/g2_cellwhisperer.py \
    --out experiments/g2_cellwhisperer --seeds 0 1 2 --batch 32
echo "G2 cellwhisperer done"
