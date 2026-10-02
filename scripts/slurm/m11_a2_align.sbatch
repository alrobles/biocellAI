#!/bin/bash
#SBATCH --job-name=a2_align
#SBATCH --output=/beegfs/a474r867/biocellai/logs/a2_align_%A.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/a2_align_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=4:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8

# M11-A2: project ROSMAP cells onto the SEA-AD 2000-HVG input space
set -euo pipefail
source /beegfs/a474r867/biocellai/venv/bin/activate
cd /beegfs/a474r867/biocellai/repos/biocellai-devel
python scripts/m11_align_genespace.py \
  --raw /beegfs/a474r867/biocellai/data/rosmap_liu2025/snRNA_multiregion_liu2025.h5ad \
  --s0 /beegfs/a474r867/biocellai/data/rosmap_s0.h5ad \
  --genes-from /beegfs/a474r867/biocellai/data/seaad_allregions_s0.h5ad \
  --out /beegfs/a474r867/biocellai/data/rosmap_for_seaad.h5ad
