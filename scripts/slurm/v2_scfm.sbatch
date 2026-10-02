#!/bin/bash
#SBATCH --job-name=v2scfm
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2scfm_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2scfm_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# R4-BENCHMARK: frozen external-scFM embedding + identical-protocol eval
# for one (cohort, seed, model, input) tuple.
#   sbatch --export=ALL,COHORT=rosmap,SEED=0,MODEL=gf_v2_104m,INPUT=native \
#          v2_scfm.sbatch
# Outputs:
#   experiments/v2_revalidation/<coh>/scfm/<model>_<input>_s<seed>.npz(.manifest.json)
#   experiments/v2_revalidation/<coh>/run_s<seed>_<model>_<input>/  (v2run schema)

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1

COHORT=${COHORT:-blood}
SEED=${SEED:-0}
MODEL=${MODEL:-gf_v1_10m}
INPUT=${INPUT:-native}
OUTBASE=${OUTBASE:-experiments/v2_revalidation}

case "$COHORT" in
  blood)      EXTRACT=$ROOT/data/tabula_blood_90000_s0.h5ad; PATH_COL="" ;;
  seaad_mtg)  EXTRACT=$ROOT/data/v2_src/seaad_mtg.h5ad;      PATH_COL=CPS_Global ;;
  rosmap)     EXTRACT=$ROOT/data/v2_src/rosmap_liu2025.h5ad; PATH_COL=path_level ;;
  *) echo "unknown COHORT $COHORT"; exit 2 ;;
esac

GF_SNAP=$(ls -d $ROOT/hf_cache/hub/models--ctheodoris--Geneformer/snapshots/*/)
SCGPT_DIR=$ROOT/data/scgpt_whole_human
CW_CKPT=$ROOT/data/cellwhisperer/cellwhisperer_clip_v1.ckpt

FOLDDIR=${OUTBASE}/${COHORT}/folds
FOLD=${FOLDDIR}/fold_s${SEED}.h5ad
FMAN=${FOLDDIR}/fold_s${SEED}.json
EMB=${OUTBASE}/${COHORT}/scfm/${MODEL}_${INPUT}_s${SEED}
RUN=${OUTBASE}/${COHORT}/run_s${SEED}_${MODEL}_${INPUT}

# completed embedding -> skip straight to eval (idempotent resubmits)
if [ -f "${EMB}.npz" ] && [ ! -f "$RUN/metrics.csv" ]; then
    EVAL_ARGS=(--emb "${EMB}.npz" --fold "$FOLD" --fold-manifest "$FMAN"
               --device cpu --out "$RUN")
    [ -n "$PATH_COL" ] && EVAL_ARGS+=(--pathology-col "$PATH_COL")
    python scripts/v2_scfm_eval.py "${EVAL_ARGS[@]}"
    exit 0
fi
[ -f "${EMB}.npz" ] && [ -f "$RUN/metrics.csv" ] && { echo "already done"; exit 0; }

python scripts/v2_scfm_embed.py \
    --extract "$EXTRACT" --fold "$FOLD" \
    --model "$MODEL" --input "$INPUT" \
    --gf-snapshot "$GF_SNAP" --scgpt-dir "$SCGPT_DIR" --cw-ckpt "$CW_CKPT" \
    --out "$EMB"

EVAL_ARGS=(--emb "${EMB}.npz" --fold "$FOLD" --fold-manifest "$FMAN"
           --device cpu --out "$RUN")
[ -n "$PATH_COL" ] && EVAL_ARGS+=(--pathology-col "$PATH_COL")
python scripts/v2_scfm_eval.py "${EVAL_ARGS[@]}"
