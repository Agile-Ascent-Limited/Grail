#!/bin/bash
# Start 8 vLLM servers for 8x A100 mining setup
# Each server runs on its own GPU and port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAIL_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_PYTHON="${GRAIL_VLLM_PYTHON:-$GRAIL_ROOT/tools/vllm-server/.venv/bin/python}"

# Model path - uses the latest downloaded checkpoint or specify manually
MODEL_PATH="${GRAIL_VLLM_MODEL_PATH:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/}"
# If MODEL_PATH has a wildcard, resolve it
if [[ "$MODEL_PATH" == *"*"* ]]; then
    RESOLVED_PATH=$(ls -d $MODEL_PATH 2>/dev/null | head -1)
    if [ -n "$RESOLVED_PATH" ]; then
        MODEL_PATH="$RESOLVED_PATH"
    else
        # Fallback to HuggingFace model ID (vLLM will download it)
        MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
    fi
fi

# Base port (servers will use 30000, 30001, ..., 30007)
BASE_PORT="${GRAIL_VLLM_BASE_PORT:-30000}"

# Number of servers to start
NUM_SERVERS="${GRAIL_VLLM_NUM_SERVERS:-8}"

# vLLM configuration
DTYPE="${GRAIL_VLLM_DTYPE:-bfloat16}"
GPU_MEMORY_UTIL="${GRAIL_VLLM_GPU_MEMORY_UTIL:-0.85}"
MAX_MODEL_LEN="${GRAIL_VLLM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${GRAIL_VLLM_MAX_NUM_SEQS:-32}"

echo "========================================"
echo "GRAIL vLLM Multi-Server Startup"
echo "========================================"
echo ""
echo "Configuration:"
echo "  vLLM Python: $VLLM_PYTHON"
echo "  Model: $MODEL_PATH"
echo "  Base port: $BASE_PORT"
echo "  Servers: $NUM_SERVERS"
echo "  GPU memory util: $GPU_MEMORY_UTIL"
echo "  Max model len: $MAX_MODEL_LEN"
echo ""

# Check if vLLM environment exists
if [ ! -f "$VLLM_PYTHON" ]; then
    echo "ERROR: vLLM Python not found at $VLLM_PYTHON"
    echo "Run: bash scripts/setup_vllm_env.sh"
    exit 1
fi

# Check if vLLM is importable
if ! "$VLLM_PYTHON" -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM not importable. Run: bash scripts/setup_vllm_env.sh"
    exit 1
fi

# Create log directory
mkdir -p /var/log/grail/vllm

# Function to start a single vLLM server
start_vllm_server() {
    local gpu_id=$1
    local port=$((BASE_PORT + gpu_id))
    local log_file="/var/log/grail/vllm/vllm-server-$gpu_id.log"

    echo "Starting vLLM server on GPU $gpu_id, port $port..."

    # Use per-GPU cache directory to avoid race conditions
    CUDA_VISIBLE_DEVICES=$gpu_id \
    VLLM_TORCH_COMPILE_CACHE="/root/.cache/vllm/torch_compile_cache_gpu$gpu_id" \
    nohup "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "grail-model" \
        --host 127.0.0.1 \
        --port "$port" \
        --dtype "$DTYPE" \
        --kv-cache-dtype auto \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --trust-remote-code \
        --enforce-eager \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "  Started with PID $pid (log: $log_file)"
    echo "$pid" > "/var/run/grail-vllm-$gpu_id.pid"
}

# Start servers for each GPU
for ((i=0; i<NUM_SERVERS; i++)); do
    start_vllm_server $i
    # Small delay between starts to avoid race conditions
    sleep 2
done

echo ""
echo "========================================"
echo "Waiting for servers to become ready..."
echo "========================================"

# Wait for all servers to be ready
TIMEOUT=300  # 5 minutes
START_TIME=$(date +%s)

for ((i=0; i<NUM_SERVERS; i++)); do
    port=$((BASE_PORT + i))
    echo -n "Checking GPU $i (port $port)..."

    while true; do
        if curl -s "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
            echo " Ready!"
            break
        fi

        ELAPSED=$(($(date +%s) - START_TIME))
        if [ $ELAPSED -gt $TIMEOUT ]; then
            echo " TIMEOUT! Check /var/log/grail/vllm/vllm-server-$i.log"
            exit 1
        fi

        sleep 2
    done
done

echo ""
echo "========================================"
echo "All vLLM servers are ready!"
echo "========================================"
echo ""
echo "Server URLs:"
for ((i=0; i<NUM_SERVERS; i++)); do
    port=$((BASE_PORT + i))
    echo "  GPU $i: http://127.0.0.1:$port"
done
echo ""
echo "To stop all servers: bash scripts/stop_vllm_servers.sh"
echo ""
