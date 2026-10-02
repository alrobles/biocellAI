#!/bin/bash
#SBATCH --job-name=m6_retrieve
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m6_retrieve_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m6_retrieve_%j.err
#SBATCH --partition=sixhour
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

DB=/beegfs/a474r867/litdump/pubmed/pubmed_full.db
mkdir -p "$ROOT/logs" data/retrieval/m6

# Step 1 (login-only pre-step if needed): HGNC aliases
if [ ! -s data/gene_aliases.json ]; then
    python scripts/fetch_gene_aliases.py \
        --markers data/manifests/markers_train_s0.json \
        --out data/gene_aliases.json || echo '{"_note":"alias fetch failed"}' > data/gene_aliases.json
fi

# Step 2: retrieval ladder over PubMed FTS5
python scripts/m6_retrieve.py \
    --db "$DB" \
    --markers data/manifests/markers_train_s0.json \
    --aliases data/gene_aliases.json \
    --outdir experiments/m6_retrieval \
    --textdir data/text \
    --docdir data/retrieval/m6 \
    --per-query-k 40 --pool-k 60

echo "M6 retrieval done. Outputs in experiments/m6_retrieval/"
