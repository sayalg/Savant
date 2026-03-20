# AF3 + HADDOCK3 Scaled Docking Pipeline

Nextflow + Slurm pipeline that scales your existing `pipeline.py` script
to ~1.2M protein-ligand jobs while staying within a 20-GPU limit.

---

## Files

```
pipeline/
├── main.nf                  # Nextflow workflow — one DOCK process per CSV row
├── nextflow.config          # Slurm resources, GPU settings, tracing
├── submit.sh                # sbatch this once; Nextflow manages everything else
├── jobs.csv                 # Your input (job_name, sequence, smiles)
└── scripts/
    ├── pipeline.py          # Your existing script — untouched
    └── check_failures.py    # Post-run failure analysis + rerun CSV generator
```

---

## Input CSV

```csv
job_name,sequence,smiles
dock1,MTVPDRSEIAGKWYVVALASNTEFFLREKDKMK...,CCCCCCCCCCCCCCCCCCCC(=O)O
dock2,MASMTGGQQMGRDPNSSS...,C1CCCCC1
...
```

---

## Running

```bash
# Submit once — Nextflow manages the full 1.2M job queue from here
sbatch submit.sh

# Watch live progress from the login node
watch -n 30 'squeue -u $USER | grep -c " R "'

# Check Nextflow's own log (tagged per job_name)
tail -f logs/nextflow_run.log

# Inspect a specific job's status
nextflow log last --fields tag,status,exit,duration | grep dock1
```

---

## After the run

```bash
# Summary + auto-generate failed_jobs.csv
python3 scripts/check_failures.py results/run_manifest.csv \
    --original-csv jobs.csv \
    --trace results/pipeline_info/trace.tsv

# Rerun only failures (Nextflow skips already-succeeded jobs via -resume)
nextflow run main.nf --input_csv failed_jobs.csv -resume
```

The manifest at `results/run_manifest.csv`:
```
job_name,status
dock1,SUCCESS
dock2,FAILED
dock3,SUCCESS
...
```

---

## GPU throttling

`maxForks 20` in the `DOCK` process is the hard ceiling on concurrent GPU jobs.
Nextflow internally queues everything else and drains it as GPUs free up.
Override at runtime without editing any file:

```bash
nextflow run main.nf --max_gpus 18   # leave headroom for others
```

The Slurm executor's `queueSize = 500` separately caps how many jobs
sit in the Slurm queue at once — avoids flooding the scheduler
while Nextflow feeds in 1.2M submissions over time.

---

## Fault tolerance

| Mechanism | Behaviour |
|---|---|
| `errorStrategy 'retry'` | Re-attempts a failed job up to `max_retries` times automatically |
| `errorStrategy 'ignore'` | After retries exhausted, marks job FAILED and continues the rest |
| `-resume` | On crash or requeue, skips every already-completed task by hash |
| `WRITE_STATUS` join | Captures failures by detecting which job_names never produced output |

---

## Runtime estimate

At ~3 min/job and 20 GPUs running in parallel:

```
1,200,000 jobs × 3 min ÷ 20 GPUs = 90,000 GPU-minutes ≈ 62.5 days wall time
```

If your cluster allows more GPUs or you can increase your allocation,
each doubling of `max_gpus` halves the wall time linearly.
