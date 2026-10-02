#!/bin/bash
#SBATCH --job-name=biocellai_m3_ablation
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m3_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m3_%j.err
# M3 — grounding ablation on full Tabula Sapiens blood (~85k cells).
# Data must be pre-downloaded to ${SCRATCH}/data/*.h5ad by the login-node
# prep step (compute nodes have no internet).

set -euo pipefail
SCRATCH=/beegfs/a474r867/biocellai
REPO=$SCRATCH/repos/biocellai-devel
VENV=$SCRATCH/venv
H5AD=${H5AD:-$SCRATCH/data/tabula_blood_90000_s0.h5ad}

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HF_HOME=$SCRATCH/hf_cache   # HF models cached on shared fs

"$VENV/bin/python" -m biocellai.experiment m2 \
    --dataset "$H5AD" \
    --seeds 0 1 2 \
    --caption-mode all \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --outdir "$SCRATCH/results/m3_ablation_85k" \
    --epochs 40 \
    --tissue blood

echo "M3 done. Artifacts:"
ls -la "$SCRATCH/results/m3_ablation_85k/"
