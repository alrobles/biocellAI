#!/bin/bash
#SBATCH --job-name=m8_sweep
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m8_sweep_%A_%a.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m8_sweep_%A_%a.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-6

# M8 sweep (Gen 0): factorial grid over the donor-level contrastive
# target, ecoreasoner-style — one config per array task, same dataset,
# same seeds, same progression suite downstream. Gates pre-registered:
# GO = rho(CPS_Global) >= 0.5 AND gap vs shuffle-null >= 0.15 AND at
# least one positive ordinal trajectory stable across both encoders.

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DATASET=$(ls -t $ROOT/data/seaad_mtg_v1c_*_s0.h5ad | head -1)

# per-task config: TAG | LABEL_KEY | DESC_JSON | EXTRA
case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=braak;     LK=braak;            DESC=data/text/m8/pathology_braak.json;     EXTRA="" ;;
  1) TAG=cerad;     LK=cerad;            DESC=data/text/m8/pathology_cerad.json;     EXTRA="" ;;
  2) TAG=thal;      LK=thal;             DESC=data/text/m8/pathology_thal.json;      EXTRA="" ;;
  3) TAG=cog;       LK=cognitive_status; DESC=data/text/m8/pathology_cognitive.json; EXTRA="" ;;
  4) TAG=cps_q3;    LK=CPS_Global;       DESC=data/text/m8/pathology_cps.json;       EXTRA="--label-bin 3" ;;
  5) TAG=cps_q4;    LK=CPS_Global;       DESC=data/text/m8/pathology_cps.json;       EXTRA="--label-bin 4" ;;
  6) TAG=combo;     LK="cell_type|adnc"; DESC=data/text/m8/combo_subclass_adnc.json; EXTRA="" ;;
esac

python -m biocellai.experiment m2 \
    --dataset "$DATASET" \
    --seeds 0 1 2 \
    --outdir "experiments/m8_sweep_${TAG}" \
    --caption-mode type_llm \
    --type-descriptions "$DESC" \
    --class-descriptions "$DESC" \
    --label-key "$LK" \
    --tissue brain \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 40 \
    --save-embeddings "experiments/m8_sweep_${TAG}/embeddings" \
    $EXTRA

echo "M8 sweep arm $TAG done"
