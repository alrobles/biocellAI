#!/bin/bash
#SBATCH --job-name=m12_s1
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m12_s1_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m12_s1_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1

# M12-S1 (ADR-009): frozen scGPT whole-human embeddings as the contrastive
# encoder input — same dual_cog_rag recipe as the C1 winner, same donor
# splits. Isolates "grounding on scFM features" vs ridge-on-features
# (0.43-0.51) and vs expression-MLP (0.604-0.607).
#   0: real labels   1: donor->pathology shuffled (null)

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DATASET=$ROOT/data/seaad_allregions_s0.h5ad
TYPEDESC=data/text/m7/type_desc_pubmed_rag_markers_alias_allregions.json
PDESC=data/text/m11/pathology_cognitive_rag.json
EMB=$ROOT/data/scgpt_emb_seaad_s0proc.npz

if [ ! -f "$EMB" ]; then
    echo "extracting aligned scGPT input emb..."
    python scripts/m12_extract_scgpt_input.py \
        --h5ad "$DATASET" \
        --emb experiments/g2_scgpt/scgpt_emb.npz \
        --out "$EMB"
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=scgptemb_dual_cog_rag;      SHUF="" ;;
  1) TAG=scgptemb_dual_cog_rag_shuf; SHUF="--shuffle-donor-labels" ;;
esac

python -m biocellai.experiment m2 \
    --dataset "$DATASET" \
    --seeds 0 1 2 \
    --outdir "experiments/m12_${TAG}" \
    --caption-mode type_llm \
    --type-descriptions "$TYPEDESC" \
    --class-descriptions "$TYPEDESC" \
    --label-key cell_type \
    --dual-head cognitive_status \
    --pathology-descriptions "$PDESC" \
    --input-emb "$EMB" \
    $SHUF \
    --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 \
    --save-embeddings "experiments/m12_${TAG}/embeddings"

python scripts/m7_progression.py \
    --emb-dir "experiments/m12_${TAG}/embeddings" \
    --donor-table "$ROOT/data/seaad_donors_allregions.csv" \
    --out "experiments/m12_${TAG}/progression"

echo "M12-S1 arm $TAG done"
