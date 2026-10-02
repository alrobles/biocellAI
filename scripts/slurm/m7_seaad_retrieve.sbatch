#!/bin/bash
#SBATCH --job-name=m7_retrieve
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m7_retrieve_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m7_retrieve_%j.err
#SBATCH --partition=sixhour
#SBATCH --time=5:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
cd "$ROOT/repos/biocellai-devel"
source "$ROOT/venv/bin/activate"

# brain curated markers (PanglaoDB coarse neural categories)
python scripts/build_panglao_map.py \
    --tsv data/panglao_markers.tsv.gz \
    --out data/manifests/panglao_brain_markers.json \
    --map brain --top 12

python scripts/m6_retrieve.py \
    --db /beegfs/a474r867/litdump/pubmed/pubmed_full.db \
    --markers data/manifests/markers_seaad_s0.json \
    --curated-markers data/manifests/panglao_brain_markers.json \
    --aliases data/gene_aliases.json \
    --context brain \
    --textdir data/text/m7 \
    --outdir experiments/m7_retrieval \
    --docdir data/retrieval/m7 \
    --arms rag_markers_alias rag_curated_alias rag_name_stripped

echo "m7 retrieval done"
