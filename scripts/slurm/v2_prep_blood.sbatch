#!/bin/bash
#SBATCH --job-name=v2prep_blood
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2prep_blood_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2prep_blood_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# V2 R1-REAL-DATA: Tabula blood (CELLxGENE). X verified as integer counts,
# donor_id/cell_type already in obs -> v2_prepare directly, no extract step.
# Blood is an identity-only cohort (no donor pathology target).

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

python scripts/v2_prepare.py \
    --input "$ROOT/data/tabula_blood_90000_s0.h5ad" \
    --counts-layer X \
    --cohort blood \
    --source-release "Tabula Sapiens blood 90k (CELLxGENE)" \
    --seeds 0 1 2 \
    --n-hvg 2000 \
    --min-genes 200 \
    --min-cells 3 \
    --out "$ROOT/repos/biocellai-devel/experiments/v2_revalidation/blood/folds"

echo "v2 blood prep done"
