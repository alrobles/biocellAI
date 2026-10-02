#!/bin/bash
#SBATCH --job-name=m4_llm_labeled
#SBATCH --partition=sixhour
#SBATCH --time=05:55:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/biocellai/logs/m4_labeled_%j.out
#SBATCH --error=/beegfs/a474r867/biocellai/logs/m4_labeled_%j.err

# M4 leakage probe: LLM is told the type name -> prose contains it.
# NOT an honest grounding result; kept as an upper-bound reference.

set -euo pipefail
ROOT=/beegfs/a474r867/biocellai
cd $ROOT/repos/biocellai-devel
export PATH=$ROOT/venv/bin:$PATH
export HF_HOME=$ROOT/hf_cache
export TOKENIZERS_PARALLELISM=false

MODEL=Qwen/Qwen2.5-14B-Instruct
DESC=$ROOT/data/text/type_desc_qwen_labeled.json
PROV=$ROOT/data/text/type_desc_qwen_labeled.prov.json
DATA=$ROOT/data/tabula_blood_90000_s0.h5ad

python scripts/m4_generate.py --variant labeled --model $MODEL \
    --markers-json data/manifests/markers_train_s0.json \
    --cell-types-json data/manifests/cell_types.json \
    --out $DESC --prov-out $PROV

python -m biocellai.experiment m2 \
    --dataset $DATA \
    --caption-mode type_llm --type-descriptions $DESC \
    --text-models all-MiniLM-L6-v2 cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --seeds 0 1 2 --epochs 40 \
    --outdir $ROOT/results/m4_llm_labeled_85k

echo "M4-labeled done. Artifacts:"
ls -la $ROOT/results/m4_llm_labeled_85k/
