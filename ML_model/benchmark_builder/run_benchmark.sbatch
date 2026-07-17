#!/bin/bash
#SBATCH --job-name=savant-benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8          # keep in sync with runtime.threads in config.yaml
#SBATCH --mem=32G
# ---- EDIT ME: cluster-specific scheduler directives ----
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
# --------------------------------------------------------
set -euo pipefail

# ---- EDIT ME: how conda is made available on your cluster ----
# e.g. `module load miniconda3` or source your conda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate savant-benchmark
# --------------------------------------------------------------

cd "$(dirname "$0")"          # run from benchmark_builder/
mkdir -p logs work output

echo "== Step 01: attach protein sequences to the training table =="
python scripts/01_add_sequences.py --config config.yaml

# Step 02 (download) needs internet. Compute nodes usually have none, so it is
# normally run on the LOGIN NODE before sbatch (see README). If your compute
# nodes DO have internet, uncomment the next line and skip the manual step.
# echo "== Step 02: download BindingDB Kd =="; python scripts/02_download_bindingdb.py --config config.yaml

echo "== Step 03: build the benchmark =="
python scripts/03_build_benchmark.py --config config.yaml

echo "== Done. Deliverables in ./output/ =="
ls -la output/
