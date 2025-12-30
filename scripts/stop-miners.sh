#!/bin/bash
# stop-miners.sh - Clean shutdown script for GRAIL miners
# Stops PM2 and kills all zombie processes
#
# Usage: ./scripts/stop-miners.sh

set -e

echo "=== GRAIL Miner Shutdown Script ==="

# Step 1: Stop PM2 processes
echo "[1/3] Stopping PM2 processes..."
pm2 stop all 2>/dev/null || true
pm2 delete all 2>/dev/null || true
sleep 1

# Step 2: Kill any zombie grail processes
echo "[2/3] Killing grail processes..."
pkill -9 -f "grail mine" 2>/dev/null || true
pkill -9 -f "grail train" 2>/dev/null || true
sleep 1

# Step 3: Kill any zombie vLLM processes
echo "[3/3] Killing vLLM processes..."
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f "VLLM::EngineCor" 2>/dev/null || true
sleep 2

# Verify cleanup
echo ""
echo "=== Cleanup complete ==="

# Check for any remaining processes
REMAINING=$(pgrep -f "grail mine" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "WARNING: Some grail processes still running: $REMAINING"
else
    echo "All grail processes stopped."
fi

REMAINING_VLLM=$(pgrep -f vllm 2>/dev/null || true)
if [ -n "$REMAINING_VLLM" ]; then
    echo "WARNING: Some vLLM processes still running: $REMAINING_VLLM"
else
    echo "All vLLM processes stopped."
fi

# Show GPU memory status
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU memory status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
fi
