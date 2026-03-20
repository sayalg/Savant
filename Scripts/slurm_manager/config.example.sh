# Configuration File for AlphaFold3-HADDOCK3 Batch Docking
# Copy this to config.sh and update paths for your system

# ==============================================================================
# SYSTEM PATHS - UPDATE THESE FOR YOUR HPC SYSTEM
# ==============================================================================

# AlphaFold3 Singularity Image
export SINGULARITY_IMAGE="/scratch/sguirale/AlphaFold3/image/alphafold3.sif"

# AlphaFold3 Model Parameters
export MODEL_PARAMS="/projects/datasets/alphafold/3.0.0/weights"

# Sequence Databases
export DATABASES="/scratch/sguirale/AlphaFold3/databases"

# AlphaFold3 Code Directory
export CODE_DIR="/scratch/sguirale/AlphaFold3/code"

# Pipeline Script Location (use absolute path)
# This should point to your pipeline.py file
export PIPELINE_SCRIPT="$(pwd)/scripts/pipeline.py"

# ==============================================================================
# SLURM CONFIGURATION
# ==============================================================================

# Partition for GPU jobs
export SLURM_PARTITION="GPU"

# GPU constraint (if needed)
export SLURM_CONSTRAINT="FP32"

# Default time limit
export SLURM_TIME="24:00:00"

# Default memory per job
export SLURM_MEMORY="32g"

# GPUs per job
export SLURM_GPUS="1"

# ==============================================================================
# JOB MANAGEMENT
# ==============================================================================

# Maximum concurrent GPU jobs (based on your cluster availability)
export MAX_GPU_JOBS=20

# Number of jobs to submit per batch
export BATCH_SIZE=10

# Seconds between status checks
export CHECK_INTERVAL=300

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

# Default starting step (mmseqs2, alphafold3, convert, extract, acpype, haddock3)
export START_FROM="mmseqs2"

# Maximum MSA sequences
export MAX_MSA_SEQS=512

# ==============================================================================
# DIRECTORY STRUCTURE
# ==============================================================================

# Scripts directory
export SCRIPTS_DIR="scripts"

# Pipeline script location
export PIPELINE_SCRIPT="scripts/pipeline.py"

# SLURM submission script
export SLURM_SCRIPT="scripts/submit_job.sh"

# Base output directory
export OUTPUT_DIR="./results"

# Job tracking database
export DB_PATH="job_tracking.db"

# ==============================================================================
# PYTHON ENVIRONMENT (uncomment and update if using conda/venv)
# ==============================================================================

# Conda environment
# export CONDA_ENV="docking_env"
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate $CONDA_ENV

# OR Virtual environment
# export VENV_PATH="/path/to/venv"
# source $VENV_PATH/bin/activate

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

# After sourcing this file, you can run:
#
# 1. Single job submission:
#    ./submit_single.sh test1 "MKTAYIAK..." "CC(=O)O"
#
# 2. Batch submission:
#    python3 master_submit.py \
#      --input-csv input.csv \
#      --singularity-image $SINGULARITY_IMAGE \
#      --model-params $MODEL_PARAMS \
#      --databases $DATABASES \
#      --code-dir $CODE_DIR \
#      --max-gpu-jobs $MAX_GPU_JOBS \
#      --batch-size $BATCH_SIZE
#
# 3. Resume batch:
#    python3 master_submit.py \
#      --input-csv input.csv \
#      --singularity-image $SINGULARITY_IMAGE \
#      --model-params $MODEL_PARAMS \
#      --databases $DATABASES \
#      --code-dir $CODE_DIR \
#      --resume
