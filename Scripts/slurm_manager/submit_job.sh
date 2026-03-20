#!/bin/bash
#SBATCH --job-name=docking
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH --constraint=FP32
#SBATCH --time=24:00:00

# SLURM Submission Script for AlphaFold3-HADDOCK3 Docking Pipeline
# 
# Arguments (passed from master_submit.py):
#   $1: Pipeline script path (absolute)
#   $2: Protein sequence (direct string)
#   $3: SMILES string
#   $4: Singularity image path
#   $5: Model parameters path
#   $6: Databases path
#   $7: Code directory path
#   $8: Output directory (job-specific)
#   $9: Job name
#   $10: Start from step (mmseqs2, alphafold3, etc.)
#   $11: Max MSA sequences

set -e  # Exit on any error

# Parse arguments
PIPELINE_SCRIPT=$1
PROTEIN_SEQ=$2
SMILES=$3
SINGULARITY_IMAGE=$4
MODEL_PARAMS=$5
DATABASES=$6
CODE_DIR=$7
OUTPUT_DIR=$8
JOB_NAME=$9
START_FROM=${10:-mmseqs2}
MAX_MSA_SEQS=${11:-512}

# Load required modules (if needed for your system)
# UNCOMMENT AND CUSTOMIZE FOR YOUR HPC:
# module purge
# module load python/3.9
# module load singularity

# Activate Python environment if using one
# UNCOMMENT AND UPDATE PATH AS NEEDED:
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate docking_env
# OR:
# source /path/to/your/venv/bin/activate

echo "=========================================="
echo "Job: $JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Started: $(date)"
echo "=========================================="

echo "Arguments:"
echo "  Pipeline script: $PIPELINE_SCRIPT"
echo "  Protein sequence length: ${#PROTEIN_SEQ}"
echo "  SMILES: $SMILES"
echo "  Singularity image: $SINGULARITY_IMAGE"
echo "  Model params: $MODEL_PARAMS"
echo "  Databases: $DATABASES"
echo "  Code dir: $CODE_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Job name: $JOB_NAME"
echo "  Start from: $START_FROM"
echo "  Max MSA seqs: $MAX_MSA_SEQS"
echo ""

# Verify pipeline script exists
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script not found: $PIPELINE_SCRIPT"
    exit 1
fi

# Verify required files and directories exist
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Singularity image not found: $SINGULARITY_IMAGE"
    exit 1
fi

if [ ! -d "$MODEL_PARAMS" ]; then
    echo "ERROR: Model parameters directory not found: $MODEL_PARAMS"
    exit 1
fi

if [ ! -d "$DATABASES" ]; then
    echo "ERROR: Databases directory not found: $DATABASES"
    exit 1
fi

if [ ! -d "$CODE_DIR" ]; then
    echo "ERROR: Code directory not found: $CODE_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Display Python version
echo "Python version:"
python3 --version
echo ""

echo "Checking dependencies:"
python3 -c "import rdkit; print(f'RDKit: {rdkit.__version__}')" 2>/dev/null || echo "RDKit not found"
python3 -c "import pandas; print(f'pandas: {pandas.__version__}')" 2>/dev/null || echo "pandas not found"
python3 -c "import Bio; print(f'BioPython: {Bio.__version__}')" 2>/dev/null || echo "BioPython not found"
python3 -c "import pymol; print('pymol: OK')" 2>/dev/null || echo "pymol not found"
echo ""

# Run the docking pipeline
echo "Starting AlphaFold3-HADDOCK3 pipeline..."
echo ""

python3 "$PIPELINE_SCRIPT" \
    -p "$PROTEIN_SEQ" \
    -s "$SMILES" \
    --singularity-image "$SINGULARITY_IMAGE" \
    --model-params "$MODEL_PARAMS" \
    --databases "$DATABASES" \
    --code-dir "$CODE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    -n "$JOB_NAME" \
    --start-from "$START_FROM" \
    --max-msa-seqs "$MAX_MSA_SEQS"

PIPELINE_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully!"
    echo "Results: $OUTPUT_DIR/csv/${JOB_NAME}_results.csv"
    
    # List output files
    echo ""
    echo "Output files created:"
    find "$OUTPUT_DIR" -type f -name "*.csv" -o -name "*.pdb" 2>/dev/null | head -20
else
    echo "Pipeline failed with exit code: $PIPELINE_EXIT_CODE"
    echo "Check error logs for details"
fi
echo "Finished: $(date)"
echo "=========================================="

exit $PIPELINE_EXIT_CODE
