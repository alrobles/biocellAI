#!/bin/bash
#SBATCH --job-name=m7_ground
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m7_ground_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m7_ground_%j.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

ARM=${ARM:-rag_markers_alias}
DESC=${DESC:-data/text/m7/type_desc_pubmed_${ARM}.json}
DATASET=${DATASET:-$(ls -t $ROOT/data/seaad_mtg_v1c_*_s0.h5ad | head -1)}

python -m biocellai.experiment m2 \
    --dataset "$DATASET" \
    --seeds 0 1 2 \
    --outdir experiments/m7_${ARM} \
    --caption-mode type_llm \
    --type-descriptions "$DESC" \
    --class-descriptions "$DESC" \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 \
    --save-embeddings "experiments/m7_${ARM}/embeddings"

echo "M7 arm $ARM done"
