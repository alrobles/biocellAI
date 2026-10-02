#!/bin/bash
#SBATCH --job-name=a2_rosmap_w
#SBATCH --output=/beegfs/a474r867/biocellai/logs/a2_rosmap_w_%A.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/a2_rosmap_w_%A.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# M11-A2: ROSMAP pathology model trained ON the SEA-AD gene space
# (rosmap_for_seaad.h5ad) so it can be applied to seaad_allregions_s0
# with zero padding.
set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

python -m biocellai.experiment m2 \
    --dataset $ROOT/data/rosmap_for_seaad.h5ad \
    --seeds 0 1 2 --outdir "experiments/a2_rosmap_path" \
    --caption-mode type_llm \
    --type-descriptions data/text/m11/pathology_rosmap.json \
    --class-descriptions data/text/m11/pathology_rosmap.json \
    --label-key pathology --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 --save-embeddings "experiments/a2_rosmap_path/embeddings"
echo "A2 rosmap-on-seaad-genes done"
