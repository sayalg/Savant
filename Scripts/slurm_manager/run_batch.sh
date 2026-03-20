#!/bin/bash
# Simple wrapper script to run batch docking with config defaults
# 
# Usage:
#   ./run_batch.sh input.csv [additional args...]
#
# Examples:
#   ./run_batch.sh input.csv
#   ./run_batch.sh input.csv --max-gpu-jobs 15
#   ./run_batch.sh input.csv --resume

set -e

# Check if config.sh exists
if [ ! -f "config.sh" ]; then
    echo "ERROR: config.sh not found!"
    echo "Create it from config.example.sh:"
    echo "  cp config.example.sh config.sh"
    echo "  nano config.sh  # Update paths"
    exit 1
fi

# Load configuration
source config.sh

# Check required argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 INPUT_CSV [additional args...]"
    echo ""
    echo "Examples:"
    echo "  $0 input.csv"
    echo "  $0 input.csv --max-gpu-jobs 15"
    echo "  $0 input.csv --resume"
    echo "  $0 input.csv --start-from haddock3"
    exit 1
fi

INPUT_CSV=$1
shift  # Remove first argument, keep the rest

# Verify input CSV exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input CSV not found: $INPUT_CSV"
    exit 1
fi

# Verify required environment variables are set
if [ -z "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: SINGULARITY_IMAGE not set in config.sh"
    exit 1
fi

if [ -z "$MODEL_PARAMS" ]; then
    echo "ERROR: MODEL_PARAMS not set in config.sh"
    exit 1
fi

if [ -z "$DATABASES" ]; then
    echo "ERROR: DATABASES not set in config.sh"
    exit 1
fi

if [ -z "$CODE_DIR" ]; then
    echo "ERROR: CODE_DIR not set in config.sh"
    exit 1
fi

echo "=========================================="
echo "Running Batch Docking Campaign"
echo "=========================================="
echo "Input CSV: $INPUT_CSV"
echo "Singularity Image: $SINGULARITY_IMAGE"
echo "Model Parameters: $MODEL_PARAMS"
echo "Databases: $DATABASES"
echo "Code Directory: $CODE_DIR"
echo "Max GPU Jobs: ${MAX_GPU_JOBS:-20}"
echo "Output Directory: ${OUTPUT_DIR:-./results}"
echo "=========================================="
echo ""

# Run master script
# Environment variables are automatically used as defaults
python3 master_submit.py \
    --input-csv "$INPUT_CSV" \
    "$@"

echo ""
echo "=========================================="
echo "Batch submission complete!"
echo "Monitor with: python3 scripts/job_utils.py stats"
echo "=========================================="
