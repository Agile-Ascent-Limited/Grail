#!/bin/bash
# =============================================================================
# 8x A100 Multi-Worker Mining Launch Script
# =============================================================================
# This script launches multiple miner workers for maximum throughput on 8x A100.
#
# USAGE:
#   ./scripts/launch_8x_a100.sh [mode]
#
# MODES:
#   8-workers  - 8 workers, 1 GPU each (best for 4B model, ~8x throughput)
#   4-workers  - 4 workers, 2 GPUs each with tensor parallel (for larger models)
#   2-workers  - 2 workers, 4 GPUs each with tensor parallel (for 30B+ models)
#
# REQUIREMENTS:
#   - Set environment variables in .env file
#   - Install flash-attn: pip install flash-attn --no-build-isolation
#
# =============================================================================

set -e

MODE=${1:-8-workers}
LOG_DIR="${GRAIL_LOG_DIR:-/var/log/grail}"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Common optimizations
export GRAIL_USE_FLASH_ATTENTION=1
export GRAIL_GENERATION_BATCH_SIZE=8

echo "=========================================="
echo "GRAIL 8x A100 Multi-Worker Mining"
echo "=========================================="
echo "Mode: $MODE"
echo "Log directory: $LOG_DIR"
echo ""

cleanup() {
    echo "Stopping all workers..."
    pkill -f "grail mine" || true
    exit 0
}

trap cleanup SIGINT SIGTERM

case $MODE in
    "8-workers")
        echo "Launching 8 workers (1 GPU each) - Best for 4B model"
        echo "------------------------------------------"
        export GRAIL_TOTAL_WORKERS=8
        export GRAIL_MULTI_GPU=0

        for i in {0..7}; do
            export GRAIL_WORKER_ID=$i
            export CUDA_VISIBLE_DEVICES=$i
            echo "Starting worker $i on GPU $i..."
            grail mine > "$LOG_DIR/worker_$i.log" 2>&1 &
            sleep 2  # Stagger startup to avoid race conditions
        done
        ;;

    "4-workers")
        echo "Launching 4 workers (2 GPUs each with tensor parallel)"
        echo "------------------------------------------"
        export GRAIL_TOTAL_WORKERS=4
        export GRAIL_MULTI_GPU=1
        export GRAIL_TENSOR_PARALLEL_SIZE=2

        for i in {0..3}; do
            export GRAIL_WORKER_ID=$i
            gpu_start=$((i * 2))
            gpu_end=$((gpu_start + 1))
            export CUDA_VISIBLE_DEVICES="$gpu_start,$gpu_end"
            echo "Starting worker $i on GPUs $gpu_start,$gpu_end..."
            grail mine > "$LOG_DIR/worker_$i.log" 2>&1 &
            sleep 2
        done
        ;;

    "2-workers")
        echo "Launching 2 workers (4 GPUs each with tensor parallel) - For 30B+ models"
        echo "------------------------------------------"
        export GRAIL_TOTAL_WORKERS=2
        export GRAIL_MULTI_GPU=1
        export GRAIL_TENSOR_PARALLEL_SIZE=4

        for i in {0..1}; do
            export GRAIL_WORKER_ID=$i
            gpu_start=$((i * 4))
            export CUDA_VISIBLE_DEVICES="$gpu_start,$((gpu_start+1)),$((gpu_start+2)),$((gpu_start+3))"
            echo "Starting worker $i on GPUs $gpu_start-$((gpu_start+3))..."
            grail mine > "$LOG_DIR/worker_$i.log" 2>&1 &
            sleep 2
        done
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: 8-workers, 4-workers, 2-workers"
        exit 1
        ;;
esac

echo ""
echo "All workers started!"
echo "Monitor logs with: tail -f $LOG_DIR/worker_*.log"
echo "Stop all workers with: pkill -f 'grail mine'"
echo ""

# Wait for all background processes
wait
