#!/bin/bash
# Simple Single Job Submission Script
# Use this for testing individual compounds

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 JOB_NAME PROTEIN_SEQUENCE SMILES [OUTPUT_DIR]"
    echo "Example: $0 test1 'MKTAYIAK...' 'CC(=O)O'"
    exit 1
fi

JOB_NAME=$1
PROTEIN_SEQ=$2
SMILES=$3
OUTPUT_DIR=${4:-./results/$JOB_NAME}

# Configuration - UPDATE FOR YOUR SYSTEM
SINGULARITY_IMAGE="/scratch/sguirale/AlphaFold3/image/alphafold3.sif"
MODEL_PARAMS="/projects/datasets/alphafold/3.0.0/weights"
DATABASES="/scratch/sguirale/AlphaFold3/databases"
CODE_DIR="/scratch/sguirale/AlphaFold3/code"

# Find pipeline script (assumes it's in scripts/ relative to current directory)
PIPELINE_SCRIPT="$(pwd)/scripts/pipeline.py"

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script not found: $PIPELINE_SCRIPT"
    echo "Make sure pipeline.py is in scripts/ directory"
    exit 1
fi

# SLURM parameters
PARTITION="GPU"
TIME="24:00:00"
MEMORY="32g"
CONSTRAINT="FP32"

echo "Submitting job: $JOB_NAME"
echo "Pipeline: $PIPELINE_SCRIPT"
echo "Output: $OUTPUT_DIR"

sbatch \
    -t "$TIME" \
    --partition="$PARTITION" \
    --mem="$MEMORY" \
    --gres=gpu:1 \
    --constraint="$CONSTRAINT" \
    -J "$JOB_NAME" \
    --output="$OUTPUT_DIR/slurm_%j.out" \
    --error="$OUTPUT_DIR/slurm_%j.err" \
    --wrap "mkdir -p $OUTPUT_DIR && python3 $PIPELINE_SCRIPT \
        -p '$PROTEIN_SEQ' \
        -s '$SMILES' \
        --singularity-image $SINGULARITY_IMAGE \
        --model-params $MODEL_PARAMS \
        --databases $DATABASES \
        --code-dir $CODE_DIR \
        --output-dir $OUTPUT_DIR \
        -n $JOB_NAME"

echo "Job submitted!"
echo "Monitor: squeue -u \$USER"
echo "Logs: tail -f $OUTPUT_DIR/slurm_*.out"
