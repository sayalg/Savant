#!/bin/bash
# submit.sh — run this once; Nextflow manages all 1.2M child jobs from here
#
#SBATCH --job-name=nf-docking
#SBATCH --output=logs/nextflow_%j.log
#SBATCH --error=logs/nextflow_%j.err
#SBATCH --time=11-00:00:00        # Nextflow head process — needs to outlive all children
#SBATCH --partition=short         # CPU partition; Nextflow itself uses almost no resources
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

module load nf-core/3.3.2
module load alphafold/3.0.0

mkdir -p logs results/pipeline_info

nextflow run main.nf \
    -resume \
    --input_csv         input/input_test.csv \
    --outdir            results \
    --pipeline_script   scripts/pipeline.py \
    --singularity_image /scratch/sguirale/AlphaFold3/image/alphafold3.sif \
    --model_params      /projects/datasets/alphafold/3.0.0/weights \
    --databases         /scratch/sguirale/AlphaFold3/databases \
    --code_dir          /scratch/sguirale/AlphaFold3/code \
    --max_gpus          20 \
    2>&1 | tee logs/nextflow_run.log

# After pipeline completes, print a quick failure summary
echo ""
echo "=== Run complete. Checking manifest ==="
python3 scripts/check_failures.py results/run_manifest.csv --original-csv input/input_test.csv
