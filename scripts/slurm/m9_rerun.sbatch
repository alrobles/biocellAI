#!/bin/bash
#SBATCH --job-name=m9_rerun
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m9_rerun_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m9_rerun_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# Gen-2: retrain the M8 sweep winners on the full-region SEA-AD build
# (11 regions, ~440k nuclei, more donors). Same protocol as m8_sweep.

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DATASET=$ROOT/data/seaad_allregions_s0.h5ad

case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=cog;   LK=cognitive_status; DESC=data/text/m8/pathology_cognitive.json ;;
  1) TAG=adnc;  LK=adnc;             DESC=data/text/m8/pathology_adnc.json ;;
  2) TAG=cerad; LK=cerad;            DESC=data/text/m8/pathology_cerad.json ;;
  3) TAG=braak; LK=braak;            DESC=data/text/m8/pathology_braak.json ;;
esac

python -m biocellai.experiment m2 \
    --dataset "$DATASET" \
    --seeds 0 1 2 \
    --outdir "experiments/m9_${TAG}" \
    --caption-mode type_llm \
    --type-descriptions "$DESC" \
    --class-descriptions "$DESC" \
    --label-key "$LK" \
    --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 \
    --save-embeddings "experiments/m9_${TAG}/embeddings"

echo "M9 arm $TAG done"
