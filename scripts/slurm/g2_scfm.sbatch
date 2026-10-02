#!/bin/bash
#SBATCH --job-name=g2_scfm
#SBATCH --output=/beegfs/a474r867/biocellai/logs/g2_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/g2_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1

# G2: zero-shot scFM embeddings on the identical SEA-AD donor split.
# task 0: Geneformer-V1-10M (30M-corpus tokenizer)
# task 1: Geneformer-V2-104M (104M-corpus tokenizer)

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

case "${SLURM_ARRAY_TASK_ID}" in
  0) M=Geneformer-V1-10M ;;
  1) M=Geneformer-V2-104M ;;
esac

python scripts/g2_geneformer.py --model "$M" \
    --out experiments/g2_geneformer --seeds 0 1 2 --batch 48
echo "G2 $M done"
