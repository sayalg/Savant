#!/bin/bash
# Simple wrapper for GPU worker-based batch docking
#
# Usage:
#   ./run_workers.sh input.csv [num_workers] [max_rows]
#
# Examples:
#   ./run_workers.sh input.csv                 # All rows, 5 workers
#   ./run_workers.sh input.csv 10              # All rows, 10 workers
#   ./run_workers.sh input.csv 5 1000          # First 1000 rows, 5 workers
#   ./run_workers.sh input.csv 5 10000         # First 10000 rows, 5 workers

set -e

if [ ! -f "config.sh" ]; then
    echo "ERROR: config.sh not found!"
    echo "Create it from config.example.sh:"
    echo "  cp config.example.sh config.sh"
    exit 1
fi

source config.sh

if [ $# -lt 1 ]; then
    echo "Usage: $0 INPUT_CSV [NUM_WORKERS] [MAX_ROWS]"
    echo ""
    echo "Examples:"
    echo "  $0 input.csv                # All rows, 5 workers"
    echo "  $0 input.csv 10             # All rows, 10 workers"
    echo "  $0 input.csv 5 1000         # First 1000 rows, 5 workers"
    echo "  $0 input.csv 5 10000        # First 10000 rows, 5 workers"
    exit 1
fi

INPUT_CSV=$1
NUM_WORKERS=${2:-${NUM_WORKERS:-5}}
MAX_ROWS=${3:-}

echo "=========================================="
echo "GPU Worker-Based Batch Docking"
echo "=========================================="
echo "Input CSV: $INPUT_CSV"
echo "Number of GPU workers: $NUM_WORKERS"
if [ -n "$MAX_ROWS" ]; then
    echo "Max rows to process: $MAX_ROWS"
else
    echo "Max rows to process: ALL"
fi
echo "Database: docking_queue.db"
echo "=========================================="
echo ""

# Step 1: Load jobs into database
echo "Step 1: Loading jobs from CSV..."
if [ -n "$MAX_ROWS" ]; then
    python3 worker_master.py load --input-csv "$INPUT_CSV" --max-rows "$MAX_ROWS"
else
    python3 worker_master.py load --input-csv "$INPUT_CSV"
fi

# Step 2: Submit workers
echo ""
echo "Step 2: Submitting $NUM_WORKERS GPU workers..."
python3 worker_master.py submit --num-workers "$NUM_WORKERS"

# Step 3: Monitor (optional - can Ctrl+C to exit)
echo ""
echo "Step 3: Monitoring progress..."
echo "Press Ctrl+C to stop monitoring (workers will continue running)"
echo ""

python3 worker_master.py monitor

echo ""
echo "=========================================="
echo "All workers submitted!"
echo "Monitor with: python3 worker_master.py status"
echo "=========================================="
