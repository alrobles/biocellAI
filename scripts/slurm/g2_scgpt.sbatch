#!/bin/bash
#SBATCH --job-name=g2_scgpt
#SBATCH --output=/beegfs/a474r867/biocellai/logs/g2_scgpt_%A.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/g2_scgpt_%A.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# G2: scGPT whole-human zero-shot embeddings on the identical split.

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

python scripts/g2_scgpt.py --out experiments/g2_scgpt --seeds 0 1 2
echo "G2 scGPT done"
