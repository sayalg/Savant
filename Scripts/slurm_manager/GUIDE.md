# Processing Large CSV Files (1.2 Million Rows)

Your CSV has 1.2 million rows, but you can't process them all at once. Here's how to handle it.

## Quick Examples

### Process First 1000 Rows
```bash
./run_workers.sh input.csv 5 1000
```

### Process First 10,000 Rows
```bash
./run_workers.sh input.csv 5 10000
```

### Process First 100,000 Rows
```bash
./run_workers.sh input.csv 10 100000
```

## Processing in Chunks

To process your 1.2M row CSV in manageable chunks:

### Option 1: Process Multiple Batches Sequentially

```bash
source config.sh

# Batch 1: Rows 0-9,999 (first 10,000)
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 0
python3 worker_master.py submit --num-workers 5
python3 worker_master.py monitor  # Wait for completion

# Batch 2: Rows 10,000-19,999 (next 10,000)
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 10000
python3 worker_master.py submit --num-workers 5
python3 worker_master.py monitor

# Batch 3: Rows 20,000-29,999 (next 10,000)
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 20000
python3 worker_master.py submit --num-workers 5
python3 worker_master.py monitor

# And so on...
```

### Option 2: Load All at Once, Process with Available GPUs

```bash
# Load first 50,000 rows
python3 worker_master.py load --input-csv huge.csv --max-rows 50000

# Submit 5 workers
python3 worker_master.py submit --num-workers 5

# Workers will process all 50,000 over time
# At 5 min/compound avg: 50,000 × 5 min = 250,000 min = 4,167 hours
# With 5 workers: 4,167 / 5 = 833 hours = 34.7 days
# BUT workers max out at 10 days...
```

**WARNING:** For very large batches, workers will hit the 10-day limit before finishing all compounds!

### Option 3: Multiple Databases for Different Batches

```bash
# Batch 1 in separate database
python3 worker_master.py load \
    --input-csv huge.csv \
    --max-rows 10000 \
    --db-path batch1.db

python3 worker_master.py submit \
    --num-workers 5 \
    --db-path batch1.db

# Batch 2 in different database
python3 worker_master.py load \
    --input-csv huge.csv \
    --max-rows 10000 \
    --skip-rows 10000 \
    --db-path batch2.db

python3 worker_master.py submit \
    --num-workers 5 \
    --db-path batch2.db

# Can run both simultaneously!
```

## Recommended Strategy for 1.2M Rows

### Strategy 1: Incremental Processing (Safest)

Process in 10,000-row chunks:
```bash
# How many batches?
# 1,200,000 / 10,000 = 120 batches

# Time per batch:
# 10,000 compounds × 5 min = 50,000 min = 833 hours
# With 5 workers: 833 / 5 = 167 hours = 7 days per batch

# Process first batch
./run_workers.sh huge.csv 5 10000

# Wait for completion, then next batch
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 10000 --overwrite
python3 worker_master.py submit --num-workers 5
```

### Strategy 2: Larger Chunks (Faster, but risky if workers crash)

Process in 50,000-row chunks:
```bash
# Batch 1: Rows 0-49,999
python3 worker_master.py load --input-csv huge.csv --max-rows 50000
python3 worker_master.py submit --num-workers 10

# Monitor closely - workers will run ~3 days
```

### Strategy 3: Continuous Loading (Most Efficient)

Let workers process while you load more:
```bash
# Load first 10,000
python3 worker_master.py load --input-csv huge.csv --max-rows 10000

# Submit workers
python3 worker_master.py submit --num-workers 5

# While workers are running, add more in batches
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 10000
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 20000

# Submit more workers if some finish early
python3 worker_master.py submit --num-workers 3
```

## Calculating Batch Sizes

**Given:**
- Average time per compound: 5 minutes
- Number of workers: 5
- Max worker runtime: 10 days = 14,400 minutes

**Maximum compounds per batch:**
```
Worker capacity = 14,400 minutes / 5 min per compound = 2,880 compounds per worker
With 5 workers = 2,880 × 5 = 14,400 compounds per 10-day batch
```

**Safe batch size:** 10,000-12,000 compounds (leaves buffer for slower compounds)

## Examples with Your 1.2M CSV

### Conservative Approach (10K chunks)
```bash
# Process first 10,000
./run_workers.sh compounds_1.2M.csv 5 10000

# Check results
python3 worker_master.py export --output batch1_results.csv

# Process next 10,000
python3 worker_master.py load \
    --input-csv compounds_1.2M.csv \
    --max-rows 10000 \
    --skip-rows 10000 \
    --overwrite

python3 worker_master.py submit --num-workers 5
```

### Aggressive Approach (50K chunks)
```bash
# Batch 1: 0-49,999
./run_workers.sh compounds_1.2M.csv 10 50000

# Batch 2: 50,000-99,999
python3 worker_master.py load \
    --input-csv compounds_1.2M.csv \
    --max-rows 50000 \
    --skip-rows 50000 \
    --overwrite

python3 worker_master.py submit --num-workers 10
```

### Parallel Processing (Multiple Databases)
```bash
# Batch 1
python3 worker_master.py load \
    --input-csv compounds_1.2M.csv \
    --max-rows 10000 \
    --db-path batch01.db

python3 worker_master.py submit \
    --num-workers 5 \
    --db-path batch01.db \
    --output-dir results_batch01

# Batch 2 (run simultaneously!)
python3 worker_master.py load \
    --input-csv compounds_1.2M.csv \
    --max-rows 10000 \
    --skip-rows 10000 \
    --db-path batch02.db

python3 worker_master.py submit \
    --num-workers 5 \
    --db-path batch02.db \
    --output-dir results_batch02

# Now you have 10 workers total across 2 batches!
```

## Command Reference

### Load Specific Rows
```bash
# First 1000
python3 worker_master.py load --input-csv input.csv --max-rows 1000

# First 10000
python3 worker_master.py load --input-csv input.csv --max-rows 10000

# Rows 10000-19999 (skip first 10000, load next 10000)
python3 worker_master.py load --input-csv input.csv --max-rows 10000 --skip-rows 10000

# Rows 100000-109999
python3 worker_master.py load --input-csv input.csv --max-rows 10000 --skip-rows 100000
```

### Using run_workers.sh
```bash
# Simple wrapper (first N rows)
./run_workers.sh input.csv 5 1000      # First 1000, 5 workers
./run_workers.sh input.csv 10 10000    # First 10000, 10 workers
./run_workers.sh input.csv 5           # All rows, 5 workers
```

## Monitoring Large Batches

### Check Progress
```bash
python3 worker_master.py status

# Output:
#   total: 10000
#   pending: 7500
#   running: 5
#   completed: 2450
#   failed: 45
```

### Estimate Time Remaining
```bash
# If 2450 completed, 7500 pending
# Average time per compound: 5 min
# 7500 × 5 min = 37,500 min = 625 hours
# With 5 workers: 625 / 5 = 125 hours = 5.2 days
```

### Export Partial Results
```bash
# Export what's done so far
python3 worker_master.py export --output partial_results.csv

# Count completed
sqlite3 docking_queue.db "SELECT COUNT(*) FROM jobs WHERE status='completed'"
```

## Best Practices for 1.2M Rows

1. **Start Small**: Test with 1000 rows first
2. **Use Conservative Batches**: 10,000-row chunks are manageable
3. **Monitor Closely**: Check progress every few hours
4. **Export Often**: Export results after each batch completes
5. **Keep Databases**: Don't delete .db files until batch is fully exported
6. **Plan for Failures**: ~1-5% failure rate is normal
7. **Stagger Submissions**: Don't submit all 1.2M at once

## Automation Script

Create a script to process all 1.2M in batches:

```bash
#!/bin/bash
# process_all_batches.sh

BATCH_SIZE=10000
NUM_BATCHES=120  # 1,200,000 / 10,000
NUM_WORKERS=5

for ((i=0; i<$NUM_BATCHES; i++)); do
    SKIP=$((i * BATCH_SIZE))
    BATCH_NUM=$((i + 1))
    
    echo "Processing batch $BATCH_NUM/$NUM_BATCHES (rows $SKIP-$((SKIP+BATCH_SIZE-1)))"
    
    # Load batch
    python3 worker_master.py load \
        --input-csv huge.csv \
        --max-rows $BATCH_SIZE \
        --skip-rows $SKIP \
        --overwrite
    
    # Submit workers
    python3 worker_master.py submit --num-workers $NUM_WORKERS
    
    # Monitor until complete
    python3 worker_master.py monitor
    
    # Export results
    python3 worker_master.py export --output "batch_${BATCH_NUM}_results.csv"
    
    echo "Batch $BATCH_NUM complete!"
    sleep 60
done

echo "All $NUM_BATCHES batches processed!"
```

## Summary

**For your 1.2M row CSV:**

✅ **Process in chunks** - Don't load all at once  
✅ **Use 10K-row batches** - Safe and manageable  
✅ **Monitor progress** - Check status regularly  
✅ **Export after each batch** - Don't lose results  
✅ **Plan for ~120 batches** - At 10K rows per batch  

**Recommended command:**
```bash
./run_workers.sh huge.csv 5 10000  # First 10K rows, 5 workers
```

**For next batch:**
```bash
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 10000 --overwrite
python3 worker_master.py submit --num-workers 5
```
