#!/bin/bash
#SBATCH --job-name=m8_path
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m8_path_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m8_path_%j.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# M8 — donor-level objective: contrast cells against their DONOR's
# pathology caption (ADNC ordinal bin) instead of the cell-type caption.
# Eval = pathology-bin classification + downstream progression regression
# on saved embeddings (m7_progression.py). Control: SHUFFLE=1 permutes
# the donor->pathology map (expected rho ~ 0).

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DATASET=${DATASET:-$(ls -t $ROOT/data/seaad_mtg_v1c_*_s0.h5ad | head -1)}
TAG=${TAG:-adnc}
EXTRA=${EXTRA:-}

python -m biocellai.experiment m2 \
    --dataset "$DATASET" \
    --seeds 0 1 2 \
    --outdir "experiments/m8_pathology_${TAG}" \
    --caption-mode type_llm \
    --type-descriptions data/text/m8/pathology_adnc.json \
    --label-key adnc \
    --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 \
    --save-embeddings "experiments/m8_pathology_${TAG}/embeddings" \
    $EXTRA

echo "M8 pathology arm $TAG done"
