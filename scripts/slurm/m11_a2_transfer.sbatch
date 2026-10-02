#!/bin/bash
#SBATCH --job-name=a2_transfer
#SBATCH --output=/beegfs/a474r867/biocellai/logs/a2_transfer_%A.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/a2_transfer_%A.err
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# M11-A2: bidirectional zero-shot transfer eval
set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

# Direction 1: SEA-AD models -> ROSMAP
for TAG in adnc cog; do
  python scripts/m11_transfer_eval.py \
    --model-dir "experiments/a2_seaad_${TAG}/embeddings" \
    --target-h5ad $ROOT/data/rosmap_for_seaad.h5ad \
    --donor-table $ROOT/data/rosmap_donors.csv \
    --target path_level \
    --ordinal-col pathology --ordinal-map data/text/m11/pathology_ordinal.json \
    --out "experiments/a2_transfer/seaad_${TAG}_to_rosmap" || echo "dir1 $TAG failed"
done

# Direction 2: ROSMAP model -> SEA-AD
python scripts/m11_transfer_eval.py \
    --model-dir experiments/a2_rosmap_path/embeddings \
    --target-h5ad $ROOT/data/seaad_allregions_s0.h5ad \
    --donor-table $ROOT/data/seaad_donors_allregions.csv \
    --target CPS_Global \
    --ordinal-col adnc --ordinal-map data/text/m11/adnc_ordinal.json \
    --out experiments/a2_transfer/rosmap_to_seaad || echo "dir2 failed"
echo "A2 transfer eval done"
