#!/bin/bash
# Automated batch processing for large CSV files
#
# Usage:
#   ./process_batches.sh INPUT_CSV BATCH_SIZE NUM_BATCHES [NUM_WORKERS]
#
# Examples:
#   ./process_batches.sh huge.csv 10000 10 5       # Process 10 batches of 10K rows with 5 workers
#   ./process_batches.sh huge.csv 5000 20 10       # Process 20 batches of 5K rows with 10 workers

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 INPUT_CSV BATCH_SIZE NUM_BATCHES [NUM_WORKERS]"
    echo ""
    echo "Examples:"
    echo "  $0 huge.csv 10000 10 5     # 10 batches of 10K rows, 5 workers"
    echo "  $0 huge.csv 5000 20 10     # 20 batches of 5K rows, 10 workers"
    echo ""
    echo "For 1.2M rows:"
    echo "  $0 huge.csv 10000 120 5    # Process all 1.2M in 120 batches"
    exit 1
fi

INPUT_CSV=$1
BATCH_SIZE=$2
NUM_BATCHES=$3
NUM_WORKERS=${4:-5}

if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input CSV not found: $INPUT_CSV"
    exit 1
fi

if [ ! -f "config.sh" ]; then
    echo "ERROR: config.sh not found"
    exit 1
fi

source config.sh

# Create results directory
RESULTS_DIR="batch_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Batch Processing Configuration"
echo "=========================================="
echo "Input CSV: $INPUT_CSV"
echo "Batch size: $BATCH_SIZE rows"
echo "Number of batches: $NUM_BATCHES"
echo "Workers per batch: $NUM_WORKERS"
echo "Total rows to process: $((BATCH_SIZE * NUM_BATCHES))"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo ""

# Log file
LOG_FILE="$RESULTS_DIR/batch_processing.log"
echo "Processing started: $(date)" | tee -a "$LOG_FILE"

# Process each batch
for ((i=0; i<$NUM_BATCHES; i++)); do
    SKIP=$((i * BATCH_SIZE))
    BATCH_NUM=$((i + 1))
    START_ROW=$SKIP
    END_ROW=$((SKIP + BATCH_SIZE - 1))
    
    echo "" | tee -a "$LOG_FILE"
    echo "=========================================="  | tee -a "$LOG_FILE"
    echo "Batch $BATCH_NUM/$NUM_BATCHES"  | tee -a "$LOG_FILE"
    echo "Rows: $START_ROW to $END_ROW"  | tee -a "$LOG_FILE"
    echo "Started: $(date)"  | tee -a "$LOG_FILE"
    echo "=========================================="  | tee -a "$LOG_FILE"
    
    # Load batch
    echo "Loading batch into database..."  | tee -a "$LOG_FILE"
    python3 worker_master.py load \
        --input-csv "$INPUT_CSV" \
        --max-rows "$BATCH_SIZE" \
        --skip-rows "$SKIP" \
        --overwrite 2>&1 | tee -a "$LOG_FILE"
    
    # Submit workers
    echo "Submitting $NUM_WORKERS workers..."  | tee -a "$LOG_FILE"
    python3 worker_master.py submit \
        --num-workers "$NUM_WORKERS" 2>&1 | tee -a "$LOG_FILE"
    
    # Monitor until complete
    echo "Monitoring progress (this will take a while)..."  | tee -a "$LOG_FILE"
    python3 worker_master.py monitor 2>&1 | tee -a "$LOG_FILE"
    
    # Export results
    BATCH_RESULTS="$RESULTS_DIR/batch_${BATCH_NUM}_results.csv"
    echo "Exporting results to $BATCH_RESULTS..."  | tee -a "$LOG_FILE"
    python3 worker_master.py export --output "$BATCH_RESULTS" 2>&1 | tee -a "$LOG_FILE"
    
    # Get statistics
    python3 worker_master.py status 2>&1 | tee -a "$LOG_FILE"
    
    echo "Batch $BATCH_NUM completed: $(date)"  | tee -a "$LOG_FILE"
    
    # Small delay between batches
    if [ $BATCH_NUM -lt $NUM_BATCHES ]; then
        echo "Waiting 60 seconds before next batch..."  | tee -a "$LOG_FILE"
        sleep 60
    fi
done

# Combine all results
echo "" | tee -a "$LOG_FILE"
echo "=========================================="  | tee -a "$LOG_FILE"
echo "Combining all batch results..."  | tee -a "$LOG_FILE"

COMBINED_FILE="$RESULTS_DIR/all_batches_combined.csv"

# Get header from first file
head -1 "$RESULTS_DIR/batch_1_results.csv" > "$COMBINED_FILE"

# Append all data (skip headers)
for ((i=1; i<=$NUM_BATCHES; i++)); do
    BATCH_FILE="$RESULTS_DIR/batch_${i}_results.csv"
    if [ -f "$BATCH_FILE" ]; then
        tail -n +2 "$BATCH_FILE" >> "$COMBINED_FILE"
        echo "Added batch $i" | tee -a "$LOG_FILE"
    else
        echo "WARNING: Batch $i results not found: $BATCH_FILE" | tee -a "$LOG_FILE"
    fi
done

echo "Combined results saved to: $COMBINED_FILE"  | tee -a "$LOG_FILE"

# Summary statistics
TOTAL_LINES=$(wc -l < "$COMBINED_FILE")
TOTAL_COMPOUNDS=$((TOTAL_LINES - 1))  # Subtract header

echo "" | tee -a "$LOG_FILE"
echo "=========================================="  | tee -a "$LOG_FILE"
echo "BATCH PROCESSING COMPLETE"  | tee -a "$LOG_FILE"
echo "=========================================="  | tee -a "$LOG_FILE"
echo "Batches processed: $NUM_BATCHES"  | tee -a "$LOG_FILE"
echo "Total compounds: $TOTAL_COMPOUNDS"  | tee -a "$LOG_FILE"
echo "Combined results: $COMBINED_FILE"  | tee -a "$LOG_FILE"
echo "Individual batches: $RESULTS_DIR/batch_*_results.csv"  | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"  | tee -a "$LOG_FILE"
echo "Finished: $(date)"  | tee -a "$LOG_FILE"
echo "=========================================="  | tee -a "$LOG_FILE"
