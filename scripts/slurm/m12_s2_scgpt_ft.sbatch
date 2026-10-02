#!/bin/bash
#SBATCH --job-name=m12_s2
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m12_s2_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m12_s2_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-17

# M12-S2 (ADR-009): end-to-end fine-tune of scGPT whole-human with the
# dual-head contrastive objective (dual_cog_rag recipe).
#   tasks  0-5 : full-FT  lr 1e-5 ep 3  x seeds{0,1,2} x {SapBERT,MiniLM}
#   tasks  6-11: lastN-2  lr 1e-4 ep 5  x seeds{0,1,2} x {SapBERT,MiniLM}
#   tasks 12-17: shuffle-null (same two ft modes) x seeds x SapBERT only
# One (seed, encoder) per task — the objective is encoder-specific.

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DATASET=$ROOT/data/seaad_allregions_s0.h5ad
SCGPT=$ROOT/data/scgpt_whole_human
TYPEDESC=data/text/m7/type_desc_pubmed_rag_markers_alias_allregions.json
PDESC=data/text/m11/pathology_cognitive_rag.json

T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 6 ]; then
    MODE=full;  LR=1e-5; EP=3;  SHUF=""
elif [ "$T" -lt 12 ]; then
    MODE=lastN; LR=1e-4; EP=5;  SHUF=""; T=$((T-6))
else
    SHUF="--shuffle-donor-labels"; T=$((T-12))
    if [ "$T" -lt 3 ]; then MODE=full;  LR=1e-5; EP=3;
    else MODE=lastN; LR=1e-4; EP=5;  T=$((T-3)); fi
fi
SEED=$((T % 3))
ENC_IDX=$((T / 3))   # shuffle tasks: T in 0-2 -> ENC_IDX=0 (SapBERT)
case "$ENC_IDX" in
  0) ENC=cambridgeltl/SapBERT-from-PubMedBERT-fulltext ;;
  *) ENC=all-MiniLM-L6-v2 ;;
esac

TAG="scgptft_${MODE}${SHUF:+_shuf}"
OUT="experiments/m12_s2/${TAG}_${ENC##*/}_s${SEED}"
EMBDIR="experiments/m12_s2/embeddings"

python scripts/m12_scgpt_ft.py \
    --dataset "$DATASET" \
    --scgpt-dir "$SCGPT" \
    --seeds "$SEED" \
    --text-models "$ENC" \
    --type-descriptions "$TYPEDESC" \
    --class-descriptions "$TYPEDESC" \
    --label-key cell_type \
    --dual-head cognitive_status \
    --pathology-descriptions "$PDESC" \
    --ft-mode "$MODE" --ft-blocks 2 --lr "$LR" --epochs "$EP" \
    --batch-size 32 --max-length 1200 --max-cells-epoch 120000 \
    $SHUF \
    --tag "$TAG" \
    --outdir "$OUT" \
    --save-embeddings "$EMBDIR"

echo "M12-S2 task ${SLURM_ARRAY_TASK_ID} done: mode=$MODE seed=$SEED enc=$ENC shuf=${SHUF:-no}"
