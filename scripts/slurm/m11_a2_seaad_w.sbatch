#!/bin/bash
#SBATCH --job-name=a2_seaad_w
#SBATCH --output=/beegfs/a474r867/biocellai/logs/a2_seaad_w_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/a2_seaad_w_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1

# M11-A2: retrain SEA-AD pathology models with weight saving (for transfer)
set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=adnc; LK=adnc;             DESC=data/text/m8/pathology_adnc.json ;;
  1) TAG=cog;  LK=cognitive_status; DESC=data/text/m8/pathology_cognitive.json ;;
esac

python -m biocellai.experiment m2 \
    --dataset $ROOT/data/seaad_allregions_s0.h5ad \
    --seeds 0 1 2 --outdir "experiments/a2_seaad_${TAG}" \
    --caption-mode type_llm --type-descriptions "$DESC" \
    --class-descriptions "$DESC" --label-key "$LK" --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 --save-embeddings "experiments/a2_seaad_${TAG}/embeddings"
echo "A2 seaad $TAG done"
