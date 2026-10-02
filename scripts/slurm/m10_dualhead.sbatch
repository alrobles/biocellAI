#!/bin/bash
#SBATCH --job-name=m10_dual
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m10_dual_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m10_dual_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# M10: hierarchical multi-head — shared encoder + cell-type caption head
# + donor-pathology caption head (the fix for the failed 96-class combo).
# Saved npz 'emb' = pathology-head projection (for progression eval);
# 'emb_type' = type-head projection. Zeroshot/probe metrics use type head.
#   tasks 0-1: real pathology heads (adnc, cog)
#   tasks 2-3: shuffle controls (donor -> pathology permuted)

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DATASET=$ROOT/data/seaad_allregions_s0.h5ad
TYPEDESC=data/text/m7/type_desc_pubmed_rag_markers_alias_allregions.json

case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=dual_adnc;  PK=adnc;             PDESC=data/text/m8/pathology_adnc.json;      SHUF="" ;;
  1) TAG=dual_cog;   PK=cognitive_status; PDESC=data/text/m8/pathology_cognitive.json; SHUF="" ;;
  2) TAG=dual_adnc_shuf; PK=adnc;             PDESC=data/text/m8/pathology_adnc.json;      SHUF="--shuffle-donor-labels" ;;
  3) TAG=dual_cog_shuf;  PK=cognitive_status; PDESC=data/text/m8/pathology_cognitive.json; SHUF="--shuffle-donor-labels" ;;
esac

python -m biocellai.experiment m2 \
    --dataset "$DATASET" \
    --seeds 0 1 2 \
    --outdir "experiments/m10_${TAG}" \
    --caption-mode type_llm \
    --type-descriptions "$TYPEDESC" \
    --class-descriptions "$TYPEDESC" \
    --label-key cell_type \
    --dual-head "$PK" \
    --pathology-descriptions "$PDESC" \
    $SHUF \
    --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 \
    --save-embeddings "experiments/m10_${TAG}/embeddings"

echo "M10 arm $TAG done"
