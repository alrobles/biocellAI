#!/bin/bash
#SBATCH --job-name=m7_download
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m7_download_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m7_download_%j.err
#SBATCH --partition=sixhour
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /beegfs/a474r867/biocellai/data
BASE=https://sea-ad-single-cell-profiling.s3.amazonaws.com

# Official SEA-AD releases (2026-06-22): identical schema, Subclass/Supertype
# labels + donor neuropathology in obs. MTG ~33GB, V1C ~16.5GB.
for R in MTG V1C; do
  F=SEAAD_${R}_RNAseq_final-nuclei.2026-06-22
  [ -s ${F}.h5ad ] || curl -L --retry 3 -o ${F}.h5ad "${BASE}/${R}/RNAseq/${F}.h5ad"
  [ -s ${F}_metadata.csv ] || curl -sL --retry 3 -o ${F}_metadata.csv "${BASE}/${R}/RNAseq/${F}_metadata.csv"
  ls -lh ${F}.h5ad ${F}_metadata.csv
done

# Allen's continuous pseudo-progression scores (donor × region)
[ -s Global_and_Local_CPS.20260105.csv ] || curl -sLO \
  "${BASE}/Multiregion_2026/model_outputs/continuous_pseudo-progression_score/Global_and_Local_CPS.20260105.csv"

echo "m7 download done"
