#!/bin/bash
#SBATCH --job-name=m11_rosmap
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m11_rosmap_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m11_rosmap_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# M11-A1: ROSMAP Liu2025 replication — single-head pathology contrastive
# (M8-style) on the 3-level ordinal Pathology label (nonAD/earlyAD/lateAD).
#   0: hard InfoNCE, real labels
#   1: soft-ordinal (ord_tau=1.0), real labels
#   2: hard, donor->pathology shuffled (null)
#   3: soft-ordinal, shuffled (null)

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DATASET=$ROOT/data/rosmap_s0.h5ad
PDESC=data/text/m11/pathology_rosmap.json

case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=rosmap_path;            SOFT="";                  SHUF="" ;;
  1) TAG=rosmap_path_soft;       SOFT="--soft-ordinal 1.0"; SHUF="" ;;
  2) TAG=rosmap_path_shuf;       SOFT="";                  SHUF="--shuffle-donor-labels" ;;
  3) TAG=rosmap_path_soft_shuf;  SOFT="--soft-ordinal 1.0"; SHUF="--shuffle-donor-labels" ;;
esac

python -m biocellai.experiment m2 \
    --dataset "$DATASET" \
    --seeds 0 1 2 \
    --outdir "experiments/m11_${TAG}" \
    --caption-mode type_llm \
    --type-descriptions "$PDESC" \
    --class-descriptions "$PDESC" \
    --label-key pathology \
    $SOFT $SHUF \
    --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 \
    --save-embeddings "experiments/m11_${TAG}/embeddings"

echo "M11 ROSMAP arm $TAG done"
