#!/bin/bash
#SBATCH --job-name=v2conf_infer
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2conf_infer_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2conf_infer_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# R6-CONFIRMATION final aggregation (single blind pass over the
# confirmation tree): transfer metrics -> long csv, then
# paired-comparison + gates for all three cohorts.
#   sbatch --dependency=afterok:<all confirmation jobs> v2_confirm_infer.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

CONF=experiments/v2_confirmation

# transfer metrics_long.csv from per-run metrics.csv
python - <<'EOF'
import pandas as pd
from pathlib import Path
rows = []
for f in sorted(Path("experiments/v2_confirmation/transfer").glob("*/s*/metrics.csv")):
    direction = f.parent.parent.name
    seed = int(f.parent.name[1:])
    df = pd.read_csv(f).assign(direction=direction, seed=seed)
    rows.append(df)
if rows:
    out = pd.concat(rows, ignore_index=True)
    Path("experiments/v2_confirmation/transfer").mkdir(exist_ok=True)
    out.to_csv("experiments/v2_confirmation/transfer/metrics_long.csv",
               index=False)
    print("transfer rows:", len(out))
else:
    print("WARNING: no transfer metrics found")
EOF

python scripts/v2_inference.py \
    --root "$CONF" \
    --cohorts seaad_mtg:CPS_Global:3,4,5 rosmap:path_level:3,4,5 seaad_pfc:CPS_Global:0,1,2 \
    --primary-cohort seaad_mtg \
    --n-boot 2000 \
    --transfer-metrics "$CONF/transfer/metrics_long.csv" \
    --out "$CONF/inference"

echo "v2 confirmation inference done -> $CONF/inference"
