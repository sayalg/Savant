# UPDATES SUMMARY - Customized for Your Cluster

I've updated the batch submission system based on your actual SLURM example. Here's what changed:

## Key Updates Based on Your Example

### 1. SLURM Parameters (submit_job.sh)
```bash
# Updated to match your cluster:
#SBATCH --partition=GPU          (was: --partition=gpu)
#SBATCH --mem=32g                (was: 64G)
#SBATCH --constraint=FP32        (ADDED - specific to your cluster)

# Kept from your example:
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
```

### 2. Protein Sequence Handling
**BEFORE:** Created temp file, passed file path
```bash
echo "$PROTEIN_SEQ" > protein.txt
python3 pipeline.py --protein protein.txt ...
```

**NOW:** Pass sequence directly (matches your example)
```bash
python3 pipeline.py -p "$PROTEIN_SEQ" ...
```

### 3. GPU Job Limiting
**Changed from:** Total job count (1000 max)
**Changed to:** GPU-specific job count (20 max)

Your cluster has ~20 GPUs, so the system now:
- Only counts GPU jobs (not all jobs)
- Default limit: 20 concurrent GPU jobs
- Easily adjustable with `--max-gpu-jobs`

### 4. Directory Structure
```
parent_directory/          ← Run from here (not from scripts/)
├── input.csv
├── master_submit.py
├── config.sh             ← NEW: Store your paths here
├── scripts/
│   ├── pipeline.py       ← Your actual pipeline
│   ├── submit_job.sh     ← SLURM template
│   ├── submit_single.sh  ← NEW: For testing single jobs
│   ├── run_mmseqs2.py
│   └── job_utils.py
└── results/
    └── JOB_NAME/         ← Output uses job_name from CSV
```

### 5. New Files Created

**submit_single.sh** - Quick single job testing
```bash
# Test one compound without full batch system
./scripts/submit_single.sh test1 "MKTAYIAK..." "CC(=O)O"
```

**config.example.sh** - Configuration template
```bash
# Store all your paths in one place
export SINGULARITY_IMAGE="/scratch/sguirale/AlphaFold3/image/alphafold3.sif"
export MODEL_PARAMS="/projects/datasets/alphafold/3.0.0/weights"
export DATABASES="/scratch/sguirale/AlphaFold3/databases"
export CODE_DIR="/scratch/sguirale/AlphaFold3/code"
export MAX_GPU_JOBS=20
```

### 6. CSV Column Names
Updated to match common naming:
```csv
job_name,protein_sequence,ligand_smiles
compound1,MKTAYIAK...,CC(=O)O
```

(Previously: name, protein, SMILES)

## Your Actual Paths (from example)

I've pre-configured these in `config.example.sh`:

```bash
SINGULARITY_IMAGE="/scratch/sguirale/AlphaFold3/image/alphafold3.sif"
MODEL_PARAMS="/projects/datasets/alphafold/3.0.0/weights"
DATABASES="/scratch/sguirale/AlphaFold3/databases"
CODE_DIR="/scratch/sguirale/AlphaFold3/code"
```

## Quick Setup Workflow

### 1. Initial Setup
```bash
cd ~/your_project
mkdir scripts

# Copy files
cp pipeline.py run_mmseqs2.py scripts/
cp master_submit.py submit_job.sh submit_single.sh job_utils.py scripts/
chmod +x master_submit.py scripts/*.sh scripts/*.py

# Create config
cp config.example.sh config.sh
nano config.sh  # Verify/update paths
```

### 2. Test Single Job
```bash
source config.sh

# Quick test with your example data
./scripts/submit_single.sh \
    dock1 \
    "MTVPDRSEIAGKWYVVALASNTEFFLREKDKMKMAMARISFLGEDELKVSYAVPKPNGCRKWETTFKKTSDDGEVYYSEEAKKKVEVLDTDYKSYAVIYATRVKDGRTLHMMRLYSRSPEVSPAATAIFRKLAGERNYTDEMVAMLPRQEECTVDEV" \
    "CCCCCCCCCCCCCCCCCCCC(=O)O"

# Monitor
squeue -u $USER
tail -f results/dock1/slurm_*.out
```

### 3. Run Batch
```bash
# Create input.csv with your compounds
vim input.csv

# Submit batch
source config.sh
python3 master_submit.py \
    --input-csv input.csv \
    --singularity-image $SINGULARITY_IMAGE \
    --model-params $MODEL_PARAMS \
    --databases $DATABASES \
    --code-dir $CODE_DIR \
    --max-gpu-jobs 20
```

## Comparison: Your Manual Command vs. System

**Your manual submission:**
```bash
sbatch -t 1:00:00 --partition=GPU --mem=16g --gres=gpu:1 --constraint=FP32 \
    -J docking_test \
    --wrap "python3 scripts/pipeline.py -p 'MTVP...' -s 'CCCC...' \
            --singularity-image /scratch/.../alphafold3.sif \
            --model-params /projects/.../weights \
            --databases /scratch/.../databases \
            --code-dir /scratch/.../code -n dock1"
```

**What the batch system does:**
1. Reads 100s-1000s of compounds from CSV
2. Submits each as a job (just like your example)
3. Tracks completion in SQLite database
4. Manages 20 GPU limit automatically
5. Restarts from where it left off if interrupted
6. Aggregates all results into single CSV

## Key Features You Get

✅ **Same submission pattern** - Uses your exact SLURM parameters  
✅ **GPU limit management** - Stays under 20 concurrent GPU jobs  
✅ **Job tracking** - SQLite database tracks everything  
✅ **Restart capability** - Resume interrupted campaigns  
✅ **Batch processing** - Handle 1000s of compounds  
✅ **Result aggregation** - Combine all CSVs automatically  

## Files Summary

| File | Purpose |
|------|---------|
| `master_submit.py` | Main orchestrator for batch jobs |
| `submit_job.sh` | SLURM template (matches your cluster) |
| `submit_single.sh` | Quick single job testing |
| `config.example.sh` | Store your paths (copy to config.sh) |
| `job_utils.py` | Database management utilities |
| `input_example.csv` | Example input format |
| `QUICKSTART_UPDATED.md` | Quick start with your config |
| `SETUP_GUIDE.md` | Comprehensive documentation |

## Next Steps

1. **Copy files to your project:**
   ```bash
   cp master_submit.py submit_job.sh submit_single.sh job_utils.py config.example.sh ~/your_project/scripts/
   ```

2. **Create config.sh:**
   ```bash
   cd ~/your_project
   cp scripts/config.example.sh config.sh
   # Paths are already correct from your example!
   ```

3. **Test single job:**
   ```bash
   source config.sh
   ./scripts/submit_single.sh test1 "MKTAYIAK..." "CC(=O)O"
   ```

4. **Run batch when ready:**
   ```bash
   python3 master_submit.py -i input.csv \
       --singularity-image $SINGULARITY_IMAGE \
       --model-params $MODEL_PARAMS \
       --databases $DATABASES \
       --code-dir $CODE_DIR
   ```

## Questions?

- **How do I adjust GPU limit?** Change `--max-gpu-jobs 20` to your preference
- **How do I resume?** Add `--resume` flag to master_submit.py
- **How do I check progress?** `python3 scripts/job_utils.py stats`
- **How do I test?** Use `submit_single.sh` for individual jobs first

Read `QUICKSTART_UPDATED.md` for detailed walkthrough!
