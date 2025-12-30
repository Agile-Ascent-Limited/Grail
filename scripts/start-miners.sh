#!/bin/bash
# start-miners.sh - Clean startup script for GRAIL miners
# Kills any zombie processes before starting PM2 miners
#
# Usage: ./scripts/start-miners.sh [ecosystem-config.js]
#   Default config: ecosystem.config.js in current directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAIL_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-ecosystem.config.js}"

echo "=== GRAIL Miner Startup Script ==="
echo "Working directory: $GRAIL_DIR"
echo "Config file: $CONFIG_FILE"
echo ""

# Change to grail directory
cd "$GRAIL_DIR"

# Step 1: Stop PM2 processes if running
echo "[1/4] Stopping any running PM2 grail processes..."
pm2 stop all 2>/dev/null || true
pm2 delete all 2>/dev/null || true
sleep 1

# Step 2: Kill any zombie grail processes
echo "[2/4] Killing zombie grail processes..."
pkill -9 -f "grail mine" 2>/dev/null || true
pkill -9 -f "grail train" 2>/dev/null || true
sleep 1

# Step 3: Kill any zombie vLLM processes
echo "[3/4] Killing zombie vLLM processes..."
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f "VLLM::EngineCor" 2>/dev/null || true
sleep 2

# Step 4: Verify GPU memory is released
echo "[4/4] Checking GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    echo ""
fi

# Start PM2 with the config
echo "=== Starting miners with PM2 ==="
if [ -f "$CONFIG_FILE" ]; then
    pm2 start "$CONFIG_FILE"
    echo ""
    echo "Miners started. Use 'pm2 logs' to monitor."
else
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -la *.config.js 2>/dev/null || echo "  No .config.js files found"
    exit 1
fi
