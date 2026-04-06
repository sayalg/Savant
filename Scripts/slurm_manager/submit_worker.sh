#!/bin/bash
#SBATCH --job-name=docking_worker
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH --constraint=FP32
#SBATCH --time=10-00:00:00

# Long-running GPU Worker for Batch Docking
# Processes compounds sequentially from a work queue
# 
# Arguments:
#   $1: Worker ID (0, 1, 2, 3, 4...)
#   $2: Database path
#   $3: Pipeline script path
#   $4: Singularity image
#   $5: Model parameters
#   $6: Databases
#   $7: Code directory
#   $8: Output directory
#   $9: Start from step
#   $10: Max MSA sequences

set -e

WORKER_ID=$1
DB_PATH=$2
PIPELINE_SCRIPT=$3
SINGULARITY_IMAGE=$4
MODEL_PARAMS=$5
DATABASES=$6
CODE_DIR=$7
OUTPUT_DIR=$8
START_FROM=${9:-mmseqs2}
MAX_MSA_SEQS=${10:-512}

# Get worker script location (same directory as this script)
SCRIPT_DIR="/scratch/sguirale/Savant/scripts"
WORKER_SCRIPT="$SCRIPT_DIR/gpu_worker.py"

echo "=========================================="
echo "GPU Worker $WORKER_ID"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "Max runtime: 10 days"
echo "=========================================="

echo "Worker ID: $WORKER_ID"
echo "Database: $DB_PATH"
echo "Pipeline: $PIPELINE_SCRIPT"
echo "Output: $OUTPUT_DIR"
echo ""

# Verify files exist
if [ ! -f "$WORKER_SCRIPT" ]; then
    echo "ERROR: Worker script not found: $WORKER_SCRIPT"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script not found: $PIPELINE_SCRIPT"
    exit 1
fi

if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found: $DB_PATH"
    exit 1
fi

# Load modules (uncomment if needed)
# module purge
# module load python/3.9

# Activate environment (uncomment if needed)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate docking_env

# Run worker
echo "Starting worker loop..."
echo "Worker will process jobs until queue is empty"
echo ""

python3 "$WORKER_SCRIPT" \
    --db-path "$DB_PATH" \
    --worker-id "$WORKER_ID" \
    --pipeline-script "$PIPELINE_SCRIPT" \
    --singularity-image "$SINGULARITY_IMAGE" \
    --model-params "$MODEL_PARAMS" \
    --databases "$DATABASES" \
    --code-dir "$CODE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --start-from "$START_FROM" \
    --max-msa-seqs "$MAX_MSA_SEQS"

WORKER_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Worker $WORKER_ID finished"
echo "Exit code: $WORKER_EXIT_CODE"
echo "Finished: $(date)"
echo "=========================================="

exit $WORKER_EXIT_CODE
