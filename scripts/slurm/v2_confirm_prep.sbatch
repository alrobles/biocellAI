#!/bin/bash
#SBATCH --job-name=v2conf_prep
#SBATCH --output=/beegfs/a474r867/bioai/logs/v2conf_prep_%x_%A.out
#SBATCH --error=/beegfs/a474r867/bioai/logs/v2conf_prep_%x_%A.err
#SBATCH --partition=sixhour
#SBATCH --time=4:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8

# R6-CONFIRMATION fold prep (frozen protocol):
#   COHORT=seaad_mtg|rosmap -> folds seeds 3,4,5 from the existing extracts
#   COHORT=seaad_pfc        -> extract SEA-AD PFC counts + folds seeds 0,1,2
#   sbatch --export=ALL,COHORT=seaad_pfc v2_confirm_prep.sbatch

set -euo pipefail
ROOT=/beegfs/a474r867/bioai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

COHORT=${COHORT:?required}
OUTBASE=experiments/v2_confirmation
FOLDS=$OUTBASE/$COHORT/folds

case "$COHORT" in
  seaad_mtg)
    EXTRACT=$ROOT/data/v2_src/seaad_mtg
    SEEDS="3 4 5"; RELEASE="SEA-AD MTG final-nuclei 2026-06-22 (UMIs layer)"
    ;;
  rosmap)
    EXTRACT=$ROOT/data/v2_src/rosmap_liu2025
    SEEDS="3 4 5"; RELEASE="ROSMAP snRNA multiregion Liu2025 (X counts)"
    ;;
  seaad_pfc)
    EXTRACT=$ROOT/data/v2_src/seaad_pfc
    SEEDS="0 1 2"; RELEASE="SEA-AD PFC final-nuclei 2026-06-22 (UMIs layer)"
    ;;
  *) echo "unknown COHORT=$COHORT"; exit 2 ;;
esac

if [ "$COHORT" = seaad_pfc ] && [ ! -f "${EXTRACT}.h5ad" ]; then
    python - <<'EOF'
import pandas as pd
d = pd.read_csv("/beegfs/a474r867/bioai/data/seaad_donors_allregions.csv")
# CPS_Global is donor-level: the allregions table only stores it on MTG rows,
# so fill it onto PFC rows from any row where it is non-null.
glob = (d.dropna(subset=["CPS_Global"]).drop_duplicates("donor_id")
        [["donor_id", "CPS_Global", "CPS_Global_ABeta", "CPS_Global_pTau"]])
pfc = d[d.region == "PFC"].drop_duplicates("donor_id")
pfc = pfc.drop(columns=[c for c in pfc.columns if c.startswith("CPS_")])
pfc = pfc.merge(glob, on="donor_id", how="left")
pfc.to_csv("/beegfs/a474r867/bioai/data/seaad_donors_pfc.csv", index=False)
print("PFC donors:", len(pfc), "with CPS_Global:", int(pfc.CPS_Global.notna().sum()))
EOF
    python scripts/v2_extract.py \
        --input "$ROOT/data/SEAAD_PFC_RNAseq_final-nuclei.2026-06-22.h5ad" \
        --counts-layer UMIs \
        --donor-col "Donor ID" \
        --rename-obs "Subclass=cell_type" \
        --keep-obs "Donor ID,Subclass,Class,Cognitive Status,Braak,CERAD score,Brain Region" \
        --donor-table "$ROOT/data/seaad_donors_pfc.csv" \
        --donor-table-key donor_id \
        --max-cells-per-donor 1500 \
        --seed 0 \
        --out "$EXTRACT"
fi

python scripts/v2_prepare.py \
    --input "${EXTRACT}.h5ad" \
    --counts-layer X \
    --cohort "$COHORT" \
    --source-release "$RELEASE" \
    --seeds $SEEDS \
    --n-hvg 2000 --min-genes 200 --min-cells 3 \
    --out "$FOLDS"

echo "v2 confirm prep $COHORT seeds $SEEDS done -> $FOLDS"
