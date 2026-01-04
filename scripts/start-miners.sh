#!/bin/bash
# start-miners.sh - Clean startup script for GRAIL miners
# Kills any zombie processes before starting PM2 miners
#
# Usage: ./scripts/start-miners.sh [ecosystem-config.js]
#   Default config: ecosystem.config.js in current directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAIL_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-current.config.js}"

echo "=== GRAIL Miner Startup Script ==="
echo "Working directory: $GRAIL_DIR"
echo "Config file: $CONFIG_FILE"
echo ""

# Change to grail directory
cd "$GRAIL_DIR"

# Step 1: Stop PM2 miner processes (keep health monitor running if it triggered this)
echo "[1/4] Stopping any running PM2 grail miner processes..."
# Stop miners individually instead of all (preserves health monitor during auto-restart)
for i in 0 1 2 3 4 5 6 7; do
    pm2 stop "grail-miner-$i" 2>/dev/null || true
    pm2 delete "grail-miner-$i" 2>/dev/null || true
done
# Also stop prefetch if running
pm2 stop grail-prefetch 2>/dev/null || true
pm2 delete grail-prefetch 2>/dev/null || true
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

# Start prefetch daemon (optional - keeps checkpoint cache warm)
if [ -f "prefetch.config.js" ]; then
    echo ""
    echo "=== Starting checkpoint prefetch daemon ==="
    pm2 start prefetch.config.js
    echo "Prefetch daemon started."
fi

# Start health monitor (skip if already running - e.g., during auto-restart)
if [ -f "scripts/health_monitor.py" ]; then
    if pm2 describe grail-health >/dev/null 2>&1; then
        echo ""
        echo "=== Health monitor already running (skipping) ==="
    else
        echo ""
        echo "=== Starting health monitor ==="
        # Use .venv python if available, otherwise system python3
        if [ -f ".venv/bin/python" ]; then
            PYTHON_PATH=".venv/bin/python"
        else
            PYTHON_PATH="python3"
        fi

        # Detect hub mode from config file or .env
        HEALTH_ARGS=""
        if grep -q "GRAIL_HUB_MODE.*['\"]1['\"]" "$CONFIG_FILE" 2>/dev/null; then
            echo "Detected HUB mode from config"
            HEALTH_ARGS="--hub"
        elif grep -qE "^(export )?GRAIL_HUB_MODE=['\"]?1['\"]?" .env 2>/dev/null; then
            echo "Detected HUB mode from .env"
            HEALTH_ARGS="--hub"
        else
            echo "Detected WORKER mode (no GRAIL_HUB_MODE found)"
        fi

        pm2 start scripts/health_monitor.py --name grail-health --interpreter "$PYTHON_PATH" -- $HEALTH_ARGS
        echo "Health monitor started."
    fi
fi
