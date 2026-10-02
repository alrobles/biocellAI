#!/bin/bash
#SBATCH --job-name=m6_curated
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m6_curated_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m6_curated_%j.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

# Fully label-free arm: PanglaoDB curated markers replace train-derived
# markers for BOTH cell captions and class captions.
python -m biocellai.experiment m2 \
    --dataset /beegfs/a474r867/biocellai/data/tabula_blood_90000_s0.h5ad \
    --seeds 0 1 2 \
    --outdir experiments/m6_curated_85k \
    --caption-mode type_markers \
    --markers-override data/manifests/panglao_markers.json \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40

echo "M6 curated-markers arm done → experiments/m6_curated_85k"
