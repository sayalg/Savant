# Quick Reference: Processing Large CSVs

## Your Situation
- CSV has **1.2 million rows**
- Can't process all at once
- Need configurable batch sizes

## Simple Commands

### Process First N Rows

```bash
# First 1000 rows
./run_workers.sh input.csv 5 1000

# First 10,000 rows
./run_workers.sh input.csv 5 10000

# First 100,000 rows
./run_workers.sh input.csv 10 100000
```

### Process Specific Range

```bash
# Rows 0-9,999 (first 10K)
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 0

# Rows 10,000-19,999 (next 10K)
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 10000

# Rows 100,000-109,999
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 100000
```

## Automated Batch Processing

Process entire 1.2M CSV in 10K chunks automatically:

```bash
# Process all 1.2M rows in 120 batches of 10K each
./process_batches.sh huge.csv 10000 120 5

# Results will be in: batch_results_YYYYMMDD_HHMMSS/
# Combined file: all_batches_combined.csv
```

## Formula for Your CSV

**1.2 million rows:**

| Batch Size | # of Batches | Command |
|------------|--------------|---------|
| 1,000 | 1,200 | `./process_batches.sh huge.csv 1000 1200 5` |
| 5,000 | 240 | `./process_batches.sh huge.csv 5000 240 5` |
| 10,000 | 120 | `./process_batches.sh huge.csv 10000 120 5` |
| 50,000 | 24 | `./process_batches.sh huge.csv 50000 24 10` |

**Recommended:** 10,000 rows per batch (120 batches total)

## Time Estimates

At 5 min/compound average:

| Batch Size | Workers | Time per Batch |
|------------|---------|----------------|
| 1,000 | 5 | ~1 hour |
| 10,000 | 5 | ~7 days |
| 10,000 | 10 | ~3.5 days |
| 50,000 | 10 | Too long! Workers time out |

**Max safe batch size:** 10,000-12,000 compounds (stays under 10-day limit)

## Manual Workflow

```bash
source config.sh

# Batch 1: First 10K
./run_workers.sh huge.csv 5 10000
python3 worker_master.py export --output batch1.csv

# Batch 2: Next 10K
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 10000 --overwrite
python3 worker_master.py submit --num-workers 5
python3 worker_master.py monitor
python3 worker_master.py export --output batch2.csv

# Batch 3: Next 10K
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 20000 --overwrite
python3 worker_master.py submit --num-workers 5
# ... and so on
```

## Automated Workflow

```bash
# Process all 1.2M in one command (will take weeks!)
./process_batches.sh huge.csv 10000 120 5

# Process smaller subset
./process_batches.sh huge.csv 10000 10 5   # Just first 100K rows
```

## Arguments Reference

### run_workers.sh
```
./run_workers.sh INPUT_CSV [NUM_WORKERS] [MAX_ROWS]

Examples:
  ./run_workers.sh data.csv 5          # All rows, 5 workers
  ./run_workers.sh data.csv 5 1000     # First 1000 rows, 5 workers
```

### worker_master.py load
```
python3 worker_master.py load --input-csv INPUT_CSV [OPTIONS]

Options:
  --max-rows N        Load only first N rows
  --skip-rows N       Skip first N rows (for chunking)
  --overwrite         Clear existing jobs first
  
Examples:
  --max-rows 1000                    # First 1000 rows
  --max-rows 10000                   # First 10000 rows
  --max-rows 10000 --skip-rows 0     # Rows 0-9,999
  --max-rows 10000 --skip-rows 10000 # Rows 10,000-19,999
```

### process_batches.sh
```
./process_batches.sh INPUT_CSV BATCH_SIZE NUM_BATCHES [NUM_WORKERS]

Examples:
  ./process_batches.sh huge.csv 10000 120 5   # 120 batches of 10K
  ./process_batches.sh huge.csv 5000 240 10   # 240 batches of 5K
```

## Tips for 1.2M Rows

✅ **Use 10K batches** - Safe, manageable size  
✅ **Start with test** - Try first 1000 rows  
✅ **Use automation** - `process_batches.sh` handles everything  
✅ **Export often** - Save results after each batch  
✅ **Monitor progress** - Check status every few hours  
✅ **Plan for ~120 batches** - At 10K per batch  

## Common Scenarios

### Test Run
```bash
./run_workers.sh huge.csv 5 100    # Just 100 rows to test
```

### Production Run (First 10K)
```bash
./run_workers.sh huge.csv 5 10000
```

### Full Automation (All 1.2M)
```bash
./process_batches.sh huge.csv 10000 120 5
# Come back in a few weeks!
```

### Custom Range
```bash
# Process rows 50,000 to 60,000
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 50000
python3 worker_master.py submit --num-workers 5
```

## Summary

**Simple:** Use `run_workers.sh` with third parameter
```bash
./run_workers.sh huge.csv 5 10000
```

**Automated:** Use `process_batches.sh` for multiple batches
```bash
./process_batches.sh huge.csv 10000 120 5
```

**Manual:** Use `--max-rows` and `--skip-rows`
```bash
python3 worker_master.py load --input-csv huge.csv --max-rows 10000 --skip-rows 0
```

Done! 🚀
