# Quick Start Guide - Updated for Your Cluster

## Your Cluster Configuration

**Partition:** GPU  
**GPU Constraint:** FP32  
**Available GPUs:** ~20  
**Max Memory:** 32g recommended  

## File Setup

```
your_project/
├── input.csv              ← Your compound list
├── master_submit.py       ← Batch orchestrator  
├── config.sh              ← Your paths (create from config.example.sh)
├── scripts/
│   ├── pipeline.py        ← Your pipeline
│   ├── submit_job.sh      ← SLURM template
│   ├── run_mmseqs2.py
│   ├── job_utils.py
│   └── submit_single.sh   ← For testing single jobs
└── results/               ← Auto-created
```

## Step 1: Initial Setup (One-time)

```bash
cd ~/your_project

# Create scripts directory
mkdir -p scripts

# Copy your files
cp /path/to/pipeline.py scripts/
cp /path/to/run_mmseqs2.py scripts/

# Make everything executable
chmod +x master_submit.py scripts/*.sh scripts/*.py

# Create config file
cp config.example.sh config.sh
nano config.sh  # Update paths
```

## Step 2: Edit config.sh

Update these critical paths in `config.sh`:

```bash
export SINGULARITY_IMAGE="/scratch/sguirale/AlphaFold3/image/alphafold3.sif"
export MODEL_PARAMS="/projects/datasets/alphafold/3.0.0/weights"
export DATABASES="/scratch/sguirale/AlphaFold3/databases"
export CODE_DIR="/scratch/sguirale/AlphaFold3/code"
export MAX_GPU_JOBS=20
```

## Step 3: Test Single Job First

**Create a test input:**
```bash
echo "job_name,protein_sequence,ligand_smiles" > test.csv
echo "test1,MTVPDRSEIAGKWYVVALASNTEFFLREKDKMKMAMARISFLGEDELKVSYAVPKPNGCRKWETTFKKTSDDGEVYYSEEAKKKVEVLDTDYKSYAVIYATRVKDGRTLHMMRLYSRSPEVSPAATAIFRKLAGERNYTDEMVAMLPRQEECTVDEV,CCCCCCCCCCCCCCCCCCCC(=O)O" >> test.csv
```

**Submit using the simple wrapper:**
```bash
source config.sh

./scripts/submit_single.sh \
    test1 \
    "MTVPDRSEIAGKWYVVALASNTEFFLREKDKMKMAMARISFLGEDELKVSYAVPKPNGCRKWETTFKKTSDDGEVYYSEEAKKKVEVLDTDYKSYAVIYATRVKDGRTLHMMRLYSRSPEVSPAATAIFRKLAGERNYTDEMVAMLPRQEECTVDEV" \
    "CCCCCCCCCCCCCCCCCCCC(=O)O"
```

**Or manually:**
```bash
sbatch -t 1:00:00 \
    --partition=GPU \
    --mem=16g \
    --gres=gpu:1 \
    --constraint=FP32 \
    -J test1 \
    --wrap "python3 scripts/pipeline.py \
        -p 'MTVPDRSEIAGKWYVVALASNTEFFLREKDKMKMAMARISFLGEDELKVSYAVPKPNGCRKWETTFKKTSDDGEVYYSEEAKKKVEVLDTDYKSYAVIYATRVKDGRTLHMMRLYSRSPEVSPAATAIFRKLAGERNYTDEMVAMLPRQEECTVDEV' \
        -s 'CCCCCCCCCCCCCCCCCCCC(=O)O' \
        --singularity-image $SINGULARITY_IMAGE \
        --model-params $MODEL_PARAMS \
        --databases $DATABASES \
        --code-dir $CODE_DIR \
        -n test1"
```

**Monitor:**
```bash
squeue -u $USER
tail -f results/test1/slurm_*.out
```

## Step 4: Run Batch Campaign

**Once test job works:**

```bash
source config.sh

python3 master_submit.py \
    --input-csv input.csv \
    --singularity-image $SINGULARITY_IMAGE \
    --model-params $MODEL_PARAMS \
    --databases $DATABASES \
    --code-dir $CODE_DIR \
    --output-dir ./results \
    --max-gpu-jobs 20 \
    --batch-size 10 \
    --check-interval 300
```

**Or create a submission script:**
```bash
cat > run_batch.sh << 'EOF'
#!/bin/bash
source config.sh

python3 master_submit.py \
    --input-csv input.csv \
    --singularity-image $SINGULARITY_IMAGE \
    --model-params $MODEL_PARAMS \
    --databases $DATABASES \
    --code-dir $CODE_DIR \
    --max-gpu-jobs $MAX_GPU_JOBS \
    --batch-size $BATCH_SIZE \
    --check-interval $CHECK_INTERVAL
EOF

chmod +x run_batch.sh
./run_batch.sh
```

## Step 5: Monitor Progress

```bash
# Statistics
python3 scripts/job_utils.py stats

# Active jobs
squeue -u $USER | grep GPU

# Log
tail -f master_script.log

# Watch in real-time
watch -n 30 'python3 scripts/job_utils.py stats'
```

## Common Commands

### Check what's running
```bash
squeue -u $USER
python3 scripts/job_utils.py stats
```

### View failed jobs
```bash
python3 scripts/job_utils.py list-failed --details
cat results/failed_job_name/slurm_*.err
```

### Reset and retry failed
```bash
python3 scripts/job_utils.py reset-failed
python3 master_submit.py ... --resume
```

### Aggregate results
```bash
python3 scripts/job_utils.py aggregate \
    --output-dir results \
    --output final_results.csv
```

### Resume interrupted campaign
```bash
python3 master_submit.py \
    --input-csv input.csv \
    --singularity-image $SINGULARITY_IMAGE \
    --model-params $MODEL_PARAMS \
    --databases $DATABASES \
    --code-dir $CODE_DIR \
    --resume  # Key flag!
```

## Input CSV Format

**Required columns (exact names):**
```csv
job_name,protein_sequence,ligand_smiles
compound1,MKTAYIAK...,CC(=O)O
compound2,MFGLATDE...,C1=CC=CC=C1
```

**Tips:**
- `job_name` must be unique (used for directory names)
- `protein_sequence` is full amino acid sequence
- `ligand_smiles` is SMILES string
- No spaces in job_name
- Keep protein sequences on one line (no line breaks)

## Resource Recommendations

### For Testing (1-3 compounds)
```bash
#SBATCH --time=1:00:00
#SBATCH --mem=16g
--max-gpu-jobs 3
--batch-size 2
```

### For Production (20 GPU limit)
```bash
#SBATCH --time=24:00:00
#SBATCH --mem=32g
--max-gpu-jobs 20
--batch-size 10
--check-interval 300
```

### For Large Proteins (>1000 residues)
```bash
#SBATCH --time=48:00:00
#SBATCH --mem=64g
```

## Troubleshooting

### Job fails with "Module not found"
- Check if you're in correct conda environment
- Verify RDKit, pandas, BioPython, pymol installed

### "Singularity image not found"
```bash
# Verify path exists
ls -lh /scratch/sguirale/AlphaFold3/image/alphafold3.sif
# Update config.sh if needed
```

### "Permission denied"
```bash
chmod +x master_submit.py scripts/*.sh scripts/*.py
```

### Jobs stuck in queue
```bash
# Check partition status
sinfo -p GPU

# Check your limits
sacctmgr show user $USER

# Reduce resource requests in submit_job.sh
```

### Database locked
```bash
# Another master_submit.py running?
ps aux | grep master_submit

# Kill old process or wait for it to finish
```

## Directory After Successful Run

```
your_project/
├── input.csv
├── job_tracking.db                    ← SQLite tracking
├── master_script.log                  ← Main log
└── results/
    ├── compound1/
    │   ├── alphafold3/
    │   ├── msa/
    │   ├── ligands/
    │   ├── topologies/
    │   ├── haddock3/
    │   ├── csv/
    │   │   └── compound1_results.csv  ← Final output
    │   ├── slurm_12345.out
    │   └── slurm_12345.err
    └── compound2/
        └── ...
```

## Advanced: Pipeline Resume Points

Skip earlier steps with `--start-from`:

```bash
# Skip MSA generation (if already done)
python3 master_submit.py ... --start-from alphafold3

# Skip structure prediction (only docking)
python3 master_submit.py ... --start-from haddock3

# Skip ligand extraction
python3 master_submit.py ... --start-from acpype
```

## Next Steps After Completion

```bash
# 1. Aggregate all results
python3 scripts/job_utils.py aggregate --output-dir results

# 2. Sort by docking score
sort -t, -k2 -n final_results.csv > sorted_results.csv

# 3. Analyze top hits
head -20 sorted_results.csv

# 4. Export job tracking
python3 scripts/job_utils.py export --output job_summary.csv
```

## Quick Reference

| Task | Command |
|------|---------|
| Submit single | `./scripts/submit_single.sh NAME SEQ SMILES` |
| Start batch | `python3 master_submit.py -i input.csv ...` |
| Resume | `python3 master_submit.py ... --resume` |
| Check status | `python3 scripts/job_utils.py stats` |
| Reset failed | `python3 scripts/job_utils.py reset-failed` |
| Aggregate | `python3 scripts/job_utils.py aggregate --output-dir results` |
| View queue | `squeue -u $USER` |

---

**Need help?** Check SETUP_GUIDE.md for detailed documentation.
