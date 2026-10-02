#!/bin/bash
#SBATCH --job-name=m9_download
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m9_download_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m9_download_%j.err
#SBATCH --partition=sixhour
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Gen-2 donor scaling: download the remaining SEA-AD regions so the
# working set covers the full vulnerability axis (EC/MEC/LEC early,
# HIP/MTG/STG/ITG/AnG/FI/PFC mid, V1C resistant). PFC region ships under
# filename DFC — saved with the PFC label to match the loader pattern.

set -euo pipefail
cd /beegfs/a474r867/biocellai/data
BASE=https://sea-ad-single-cell-profiling.s3.amazonaws.com
REL=2026-06-22

for R in AnG HIP ITG MEC LEC STG FI; do
  F=SEAAD_${R}_RNAseq_final-nuclei.${REL}
  [ -s ${F}.h5ad ] || curl -L --retry 3 -o ${F}.h5ad "${BASE}/${R}/RNAseq/${F}.h5ad"
  ls -lh ${F}.h5ad
done

# PFC region: object lives under PFC/ prefix but is named DFC
F=SEAAD_PFC_RNAseq_final-nuclei.${REL}
[ -s ${F}.h5ad ] || curl -L --retry 3 -o ${F}.h5ad \
  "${BASE}/PFC/RNAseq/SEAAD_DFC_RNAseq_final-nuclei.${REL}.h5ad"
ls -lh ${F}.h5ad

echo "m9 download done"
