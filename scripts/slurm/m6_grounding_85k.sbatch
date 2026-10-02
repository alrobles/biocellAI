#!/bin/bash
#SBATCH --job-name=m6_ground
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m6_ground_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m6_ground_%j.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

ARM=${ARM:-rag_markers}
DESC=${DESC:-data/text/type_desc_pubmed_${ARM}.json}

python -m biocellai.experiment m2 \
    --dataset /beegfs/a474r867/biocellai/data/tabula_blood_90000_s0.h5ad \
    --seeds 0 1 2 \
    --outdir experiments/m6_${ARM}_85k \
    --caption-mode type_llm \
    --type-descriptions "$DESC" \
    --class-descriptions "$DESC" \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40

echo "M6 arm $ARM done → experiments/m6_${ARM}_85k"
