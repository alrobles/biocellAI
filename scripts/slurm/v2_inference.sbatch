#!/bin/bash
#SBATCH --job-name=v2infer
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2infer_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2infer_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# V2 R6-INFERENCE aggregation: donor-clustered paired bootstrap +
# permutation-null gates -> inference/paired_comparisons.csv + gates.json
#   sbatch v2_inference.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1

OUTBASE=${OUTBASE:-experiments/v2_revalidation}

python scripts/v2_inference.py \
    --root "$OUTBASE" \
    --cohorts seaad_mtg:CPS_Global rosmap:path_level \
    --primary-cohort seaad_mtg \
    --n-boot 2000 \
    --transfer-metrics "$OUTBASE/transfer/metrics_long.csv" \
    --out "$OUTBASE/inference"

echo "v2 inference done -> $OUTBASE/inference"
