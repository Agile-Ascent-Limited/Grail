#!/bin/bash
# Check status of all vLLM servers

BASE_PORT="${GRAIL_VLLM_BASE_PORT:-30000}"
NUM_SERVERS="${GRAIL_VLLM_NUM_SERVERS:-8}"

echo "========================================"
echo "vLLM Server Status"
echo "========================================"
echo ""

all_ready=true

for ((i=0; i<NUM_SERVERS; i++)); do
    port=$((BASE_PORT + i))
    pidfile="/var/run/grail-vllm-$i.pid"

    # Check PID
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            pid_status="PID $pid"
        else
            pid_status="PID $pid (dead)"
            all_ready=false
        fi
    else
        pid_status="No PID file"
        all_ready=false
    fi

    # Check HTTP
    if curl -s "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        http_status="OK"
    else
        http_status="FAILED"
        all_ready=false
    fi

    printf "GPU %d (port %d): HTTP=%-6s %s\n" "$i" "$port" "$http_status" "$pid_status"
done

echo ""

if $all_ready; then
    echo "All servers are ready!"
else
    echo "Some servers are not ready. Check logs in /var/log/grail/vllm/"
fi

# Show GPU memory usage
echo ""
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s / %s MiB\n", $1, $2, $3}'
