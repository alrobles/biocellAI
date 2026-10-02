#!/bin/bash
#SBATCH --job-name=m6_synth
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m6_synth_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m6_synth_%j.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:l40:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

VARIANT=${VARIANT:-honest}
DOCDIR=${DOCDIR:-data/retrieval/m6/rag_markers_alias}
DOCTAG=${DOCTAG:-$(basename "$DOCDIR")}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
MODELTAG=${MODELTAG:-qwen}
CONTEXT=${CONTEXT:-blood}
MARKERS=${MARKERS:-data/manifests/markers_train_s0.json}
EXTRA=${EXTRA:-}

python scripts/m6_synthesize.py \
    --variant "$VARIANT" \
    --docdir "$DOCDIR" \
    --markers-json "$MARKERS" \
    --model "$MODEL" \
    --context "$CONTEXT" \
    $EXTRA \
    --out "data/text/type_desc_ragllm_${VARIANT}_${DOCTAG}_${MODELTAG}.json" \
    --prov-out "data/text/type_desc_ragllm_${VARIANT}_${DOCTAG}_${MODELTAG}.prov.json"

echo "synthesis $VARIANT done"
