#!/bin/bash
# Stop all vLLM servers started by start_vllm_servers.sh

echo "Stopping all vLLM servers..."

# Kill by PID files
for pidfile in /var/run/grail-vllm-*.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping PID $pid ($(basename $pidfile))..."
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

# Also kill any remaining vLLM processes (backup)
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

echo "All vLLM servers stopped."

# Wait for GPU memory to be freed
if command -v nvidia-smi &> /dev/null; then
    echo "Waiting for GPU memory to be freed..."
    sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv
fi
