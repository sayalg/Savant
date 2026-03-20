# AlphaFold3-HADDOCK3 Batch Docking System
## Setup and Usage Guide

This system manages large-scale protein-ligand docking campaigns on SLURM clusters with GPU resource management, job tracking, and automatic restart capability.

## Directory Structure

```
your_project/                    (run from here)
├── input.csv                    (your input file)
├── master_submit.py             (orchestration script)
├── job_tracking.db              (created automatically)
├── master_script.log            (created automatically)
├── scripts/                     (your scripts directory)
│   ├── pipeline.py              (main docking pipeline)
│   ├── submit_job.sh            (SLURM template)
│   ├── run_mmseqs2.py           (MSA generation)
│   └── job_utils.py             (database utilities)
└── results/                     (created automatically)
    ├── compound1/
    │   ├── alphafold3/
    │   ├── msa/
    │   ├── ligands/
    │   ├── topologies/
    │   ├── haddock3/
    │   ├── csv/
    │   │   └── compound1_results.csv
    │   ├── slurm_12345.out
    │   └── slurm_12345.err
    ├── compound2/
    └── ...
```

## Step 1: Prepare Your Environment

### 1.1 Create Directory Structure

```bash
# Create your project directory
mkdir -p ~/my_docking_project
cd ~/my_docking_project

# Create scripts subdirectory
mkdir -p scripts

# Copy your files
cp /path/to/pipeline.py scripts/
cp /path/to/run_mmseqs2.py scripts/
cp /path/to/master_submit.py .
cp /path/to/submit_job.sh scripts/
cp /path/to/job_utils.py scripts/
```

### 1.2 Make Scripts Executable

```bash
chmod +x master_submit.py
chmod +x scripts/submit_job.sh
chmod +x scripts/pipeline.py
chmod +x scripts/job_utils.py
```

### 1.3 Customize SLURM Script

Edit `scripts/submit_job.sh` for your HPC system:

```bash
nano scripts/submit_job.sh
```

**Key sections to customize:**

**Lines 3-9: Resource requirements**
```bash
#SBATCH --partition=gpu        # Your GPU partition name
#SBATCH --cpus-per-task=8      # CPUs per job
#SBATCH --mem=64G              # Memory per job
#SBATCH --gres=gpu:1           # GPUs per job
#SBATCH --time=48:00:00        # Max runtime
```

**Lines 29-33: Module loading**
```bash
module purge
module load python/3.9         # Your Python module
module load singularity/3.8    # Your Singularity module
module load cuda/11.8          # Your CUDA module
```

**Lines 36-39: Python environment (if using conda/venv)**
```bash
# Uncomment and update:
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate docking_env
# OR
# source /path/to/venv/bin/activate
```

## Step 2: Prepare Input CSV

Create `input.csv` in your project directory:

```csv
job_name,protein_sequence,ligand_smiles
compound1,MKTAYIAK...,CC(=O)O
compound2,MFGLATDE...,C1=CC=CC=C1
compound3,MDKIIHLT...,CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

**Column requirements:**
- `job_name`: Unique identifier (used for output directories)
- `protein_sequence`: Full amino acid sequence
- `ligand_smiles`: SMILES string for the ligand

**Example file:** See `input_example.csv`

## Step 3: Test with Small Batch

**IMPORTANT: Always test with 2-3 compounds first!**

```bash
# Create test CSV
head -4 input.csv > test_input.csv

# Run test batch (adjust paths for your system)
python3 master_submit.py \
  --input-csv test_input.csv \
  --singularity-image /path/to/alphafold3.sif \
  --model-params /path/to/af3_models \
  --databases /path/to/databases \
  --code-dir /path/to/alphafold3_code \
  --output-dir ./test_results \
  --max-gpu-jobs 3 \
  --batch-size 2 \
  --check-interval 120
```

**Monitor test jobs:**
```bash
# Watch queue
watch squeue -u $USER

# Check logs
tail -f master_script.log

# View job statistics
python3 scripts/job_utils.py stats

# Check SLURM output
cat test_results/compound1/slurm_*.out
cat test_results/compound1/slurm_*.err
```

## Step 4: Run Full Campaign

Once test jobs complete successfully:

```bash
python3 master_submit.py \
  --input-csv input.csv \
  --singularity-image /path/to/alphafold3.sif \
  --model-params /path/to/af3_models \
  --databases /path/to/databases \
  --code-dir /path/to/alphafold3_code \
  --output-dir ./results \
  --max-gpu-jobs 20 \
  --batch-size 10 \
  --check-interval 300
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--input-csv` | Input CSV file | `input.csv` |
| `--singularity-image` | AF3 singularity image | `/data/af3.sif` |
| `--model-params` | AF3 model parameters | `/data/models` |
| `--databases` | Sequence databases | `/data/databases` |
| `--code-dir` | AF3 code directory | `/data/alphafold3` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--slurm-script` | `scripts/submit_job.sh` | SLURM template |
| `--output-dir` | `./results` | Output directory |
| `--db-path` | `job_tracking.db` | SQLite database |
| `--max-gpu-jobs` | `20` | Max concurrent GPU jobs |
| `--batch-size` | `10` | Jobs per batch |
| `--check-interval` | `300` | Seconds between checks |
| `--start-from` | `mmseqs2` | Pipeline start step |
| `--max-msa-seqs` | `512` | Max MSA sequences |
| `--resume` | False | Resume from database |
| `--overwrite` | False | Overwrite existing jobs |

### Pipeline Start Steps

Use `--start-from` to resume from a specific step:

- `mmseqs2`: Generate MSA (default, full pipeline)
- `alphafold3`: Skip MSA, run AF3
- `convert`: Skip AF3, convert CIF to PDB
- `extract`: Skip structure prediction, extract ligand
- `acpype`: Skip extraction, generate topology
- `haddock3`: Skip all preparation, run docking only

**Example: Resume from HADDOCK3**
```bash
python3 master_submit.py ... --start-from haddock3
```

## GPU Resource Management

The system automatically manages GPU job limits:

1. **Checks active GPU jobs** using `squeue`
2. **Waits if at limit** before submitting more
3. **Submits in batches** to avoid overwhelming scheduler

**Configure for your HPC:**
```bash
# For 20 GPUs available
--max-gpu-jobs 20

# For 50 GPUs available
--max-gpu-jobs 50

# Conservative start (test)
--max-gpu-jobs 5
```

**Batch size recommendations:**
- 20 GPUs: `--batch-size 10`
- 50 GPUs: `--batch-size 20`
- 100+ GPUs: `--batch-size 50`

## Monitoring and Management

### Check Progress

```bash
# View statistics
python3 scripts/job_utils.py stats

# List failed jobs
python3 scripts/job_utils.py list-failed --details

# Check active jobs
squeue -u $USER

# Watch log
tail -f master_script.log
```

### Handle Failed Jobs

```bash
# Reset all failed jobs to pending
python3 scripts/job_utils.py reset-failed

# Reset specific jobs
python3 scripts/job_utils.py reset --jobs compound1 compound2

# Delete jobs from tracking
python3 scripts/job_utils.py delete --jobs test_compound
```

### Resume Interrupted Campaign

If master script stops (Ctrl+C, connection loss, etc.):

```bash
# Simply resume with --resume flag
python3 master_submit.py \
  --input-csv input.csv \
  --singularity-image /path/to/af3.sif \
  --model-params /path/to/models \
  --databases /path/to/databases \
  --code-dir /path/to/code \
  --resume  # Don't reload CSV, use existing database
```

### Aggregate Results

```bash
# Combine all result CSVs
python3 scripts/job_utils.py aggregate \
  --output-dir ./results \
  --output final_results.csv

# Export job tracking
python3 scripts/job_utils.py export \
  --output job_tracking_export.csv

# Export only completed jobs
python3 scripts/job_utils.py export \
  --output completed_jobs.csv \
  --status completed
```

## Troubleshooting

### Jobs Fail Immediately

**Check SLURM logs:**
```bash
cat results/compound1/slurm_*.err
```

**Common issues:**
1. **Module not found**: Update module names in `submit_job.sh`
2. **Python dependencies missing**: Install RDKit, pandas, BioPython, pymol
3. **Wrong paths**: Verify all paths are absolute and correct
4. **Insufficient memory**: Increase `#SBATCH --mem=` in submit_job.sh

### Jobs Stuck in Queue

**Check partition and resources:**
```bash
sinfo                           # View partitions
squeue -u $USER                 # Your queued jobs
scontrol show job <JOB_ID>      # Job details
```

**Possible causes:**
- Requested more resources than available
- Partition is full
- Wrong partition name

### Database Corruption

**Export current state:**
```bash
python3 scripts/job_utils.py export --output backup.csv
```

**Start fresh:**
```bash
rm job_tracking.db
python3 master_submit.py ... --overwrite
```

### Output Files Missing

**Find jobs marked complete but missing outputs:**
```bash
python3 scripts/job_utils.py find-missing --output-dir ./results
```

**Reset those jobs:**
```bash
python3 scripts/job_utils.py reset --jobs <job_name1> <job_name2>
```

## Performance Tuning

### Small Dataset (<100 compounds)
```bash
--max-gpu-jobs 10
--batch-size 5
--check-interval 180
```

### Medium Dataset (100-500 compounds)
```bash
--max-gpu-jobs 20
--batch-size 10
--check-interval 300
```

### Large Dataset (500+ compounds)
```bash
--max-gpu-jobs 20
--batch-size 15
--check-interval 600
```

### Check Interval Guidelines

- **Fast jobs (<2 hours)**: 120-180 seconds
- **Medium jobs (2-6 hours)**: 300-600 seconds
- **Long jobs (>6 hours)**: 600-900 seconds

## Output Structure

Each job creates:
```
results/compound1/
├── alphafold3/
│   └── predictions/
│       └── compound1/
│           └── compound1_model.pdb
├── msa/
│   └── compound1_alignment.a3m
├── ligands/
│   └── compound1_ligand.pdb
├── topologies/
│   └── compound1_ligand/
│       └── compound1_ligand.acpype/
├── haddock3/
│   └── compound1/
│       └── output/
│           └── analysis/
├── csv/
│   └── compound1_results.csv  ← FINAL OUTPUT
├── slurm_12345.out
└── slurm_12345.err
```

## Final Results CSV

Contains:
- **Job ID**: Job name
- **Ligand properties**: MW, LogP, TPSA, H-donors, H-acceptors, etc.
- **Docking metrics**: Score, VDW, electrostatic, desolvation, BSA

**View results:**
```bash
# With pandas
python3 -c "import pandas as pd; df = pd.read_csv('final_results.csv'); print(df)"

# With csvkit
csvlook final_results.csv | less

# Sort by score
sort -t, -k2 -n final_results.csv | head -20
```

## Best Practices

1. **Test first**: Always run 2-3 compounds before full campaign
2. **Conservative limits**: Start with lower `--max-gpu-jobs`, increase gradually
3. **Monitor logs**: Check `master_script.log` and SLURM outputs regularly
4. **Export tracking**: Periodically save database to CSV for backup
5. **Document paths**: Keep record of singularity image, model, database paths
6. **Clean failed jobs**: Review and retry failed jobs after fixing issues

## Example Workflow

```bash
# 1. Setup
cd ~/my_docking_project
cp pipeline.py run_mmseqs2.py scripts/

# 2. Prepare input
vim input.csv  # Create your compound list

# 3. Test
head -4 input.csv > test.csv
python3 master_submit.py --input-csv test.csv ... --max-gpu-jobs 3

# 4. Monitor test
watch squeue -u $USER
tail -f master_script.log

# 5. Run full campaign
python3 master_submit.py --input-csv input.csv ... --max-gpu-jobs 20

# 6. Check progress
python3 scripts/job_utils.py stats

# 7. Handle failures
python3 scripts/job_utils.py list-failed --details
python3 scripts/job_utils.py reset-failed

# 8. Aggregate results
python3 scripts/job_utils.py aggregate --output-dir results
```

## Support

For issues:
1. Check `master_script.log`
2. Check SLURM logs: `results/*/slurm_*.err`
3. Verify all paths are correct and absolute
4. Ensure dependencies are installed
5. Contact your HPC admin for SLURM configuration help

Good luck with your docking campaign! 🚀
