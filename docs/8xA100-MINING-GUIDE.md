# GRAIL Mining Guide: 8x A100 80GB Setup

This guide covers running the GRAIL miner on a server with 8x NVIDIA A100 80GB GPUs using PM2 for process management.

## Architecture Overview

For maximum throughput on 8x A100 with a **4B model** (current default), we run **8 parallel workers**, each using 1 GPU. This provides ~8x throughput compared to single-GPU mining.

```
┌─────────────────────────────────────────────────────────────────┐
│                     8x A100 Server                              │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┤
│ Worker 0│ Worker 1│ Worker 2│ Worker 3│ Worker 4│ Worker 5│ ... │
│  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │  GPU 4  │  GPU 5  │ ... │
├─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────┤
│              Shared: Checkpoint Cache, R2 Storage               │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Install System Dependencies

```bash
# CUDA drivers (should be pre-installed on cloud instances)
nvidia-smi  # Verify GPUs are visible

# Git
sudo apt-get install -y git

# Node.js and PM2
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g pm2
```

### 2. Clone the Repository

```bash
git clone https://github.com/one-covenant/grail
cd grail
```

### 3. Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

### 4. Install Python Dependencies

Follow the official installation from [docs/miner.md](miner.md):

```bash
cd /root/Grail

# Create venv with Python 3.10 (required by grail) and install
uv venv --python 3.10 && source .venv/bin/activate
uv sync

# Install bittensor CLI (for wallet management)
uv pip install bittensor-cli

# Install orjson for faster JSON (3-5x faster)
uv pip install orjson

# Install Flash Attention 2 (~30% faster) - builds from source, takes 10-15 min
uv pip install flash-attn --no-build-isolation

# Verify flash-attn installed
python -c "import flash_attn; print(flash_attn.__version__)"
```

**Notes:**
- Flash Attention builds from source and takes 10-15 minutes - be patient
- If flash-attn fails, mining still works without it (just ~30% slower)
- Do NOT install bitsandbytes - it breaks torch version compatibility

### 4b. Optional: vLLM Backend (5-10x faster inference)

vLLM provides significantly faster inference but requires an isolated environment due to dependency conflicts with bittensor.

```bash
# Run the setup script (creates isolated env at tools/vllm-server/.venv/)
bash scripts/setup_vllm_env.sh

# Verify installation
tools/vllm-server/.venv/bin/python -c "import vllm; print(vllm.__version__)"
```

After setup, add to your `.env`:
```bash
GRAIL_VLLM_PYTHON=/root/Grail/tools/vllm-server/.venv/bin/python
```

**Note:** Each miner automatically spawns its own vLLM server on port `30000 + worker_id`. No manual server management needed.

**Performance comparison:**

| Backend | Throughput | Setup Complexity |
|---------|-----------|------------------|
| HuggingFace | 1x (baseline) | None |
| Flash Attention | ~1.3x | Medium (build from source) |
| vLLM | 5-10x | Easy (isolated env) |

**Note:** vLLM includes flash-attn built-in, so you don't need to build flash-attn separately if using vLLM.

### 5. Wallet Setup

**Regenerate existing wallet from mnemonic (if migrating to new server):**
```bash
# Regenerate coldkey from mnemonic
btcli w regen_coldkey --wallet-name YOUR_WALLET --wallet_path "~/.bittensor/wallets/" --mnemonic "your 12 word mnemonic phrase here" --overwrite

# Regenerate hotkey from mnemonic
btcli w regen_hotkey --wallet-name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY --wallet_path "~/.bittensor/wallets/" --mnemonic "your 12 word mnemonic phrase here" --overwrite
```

**Or create new wallet:**
```bash
btcli wallet new_coldkey --wallet.name YOUR_WALLET
btcli wallet new_hotkey --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY
```

**Register on subnet (if not already done):**
```bash
btcli subnet register --netuid 81 --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY
```

**Verify wallet is configured:**
```bash
btcli w list
# Should show your coldkey and hotkey
```

---

## Verify Setup (Optional)

Before starting production mining, you can run the benchmark to test throughput:

```bash
# Set wallet environment variables
export BT_WALLET_COLD=your_coldkey_name
export BT_WALLET_HOT=your_hotkey_name

# Run benchmark (requires wallet for GRAIL proofs)
GRAIL_USE_FLASH_ATTENTION=1 python scripts/benchmark_throughput.py --num-problems 5 --workers 8
```

Expected output shows rollouts/second per worker. With 8 workers, multiply by 8 for total throughput.

See [THROUGHPUT-BENCHMARK.md](THROUGHPUT-BENCHMARK.md) for detailed benchmark options.

---

## Environment Configuration

Create a `.env` file in the grail directory:

```bash
# =============================================================================
# GRAIL 8x A100 MINER CONFIGURATION
# =============================================================================

# Network
BT_NETWORK=finney
NETUID=81

# Wallet
BT_WALLET_COLD=your_coldkey_name
BT_WALLET_HOT=your_hotkey_name

# R2 Storage (required)
R2_ACCOUNT_ID=your_account_id
R2_BUCKET_ID=your_bucket_id
R2_WRITE_ACCESS_KEY_ID=your_write_key
R2_WRITE_SECRET_ACCESS_KEY=your_write_secret
R2_READ_ACCESS_KEY_ID=your_read_key
R2_READ_SECRET_ACCESS_KEY=your_read_secret

# =============================================================================
# PERFORMANCE OPTIMIZATION (8x A100 RECOMMENDED SETTINGS)
# =============================================================================

# Flash Attention 2 - ~30% faster inference
GRAIL_USE_FLASH_ATTENTION=1

# Generation batch size - how many rollouts per batch
# For A100 80GB with 4B model: 8 is optimal
GRAIL_GENERATION_BATCH_SIZE=8

# Upload bandwidth (adjust based on your network)
# Datacenter: 1000-10000, Home: 50-200
GRAIL_UPLOAD_BANDWIDTH_MBPS=1000

# Checkpoint cache directory (use fast NVMe if available)
GRAIL_CACHE_DIR=/var/cache/grail

# =============================================================================
# MULTI-WORKER CONFIGURATION
# =============================================================================
# These are set per-worker via PM2 ecosystem file (see below)
# GRAIL_WORKER_ID=0
# GRAIL_TOTAL_WORKERS=8
# CUDA_VISIBLE_DEVICES=0

# =============================================================================
# OPTIONAL: WANDB MONITORING
# =============================================================================
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=grail-mining
WANDB_ENTITY=your_team
GRAIL_MONITORING_BACKEND=wandb
```

---

## PM2 Ecosystem Configuration

Create `ecosystem.config.js` in the grail directory:

```javascript
// ecosystem.config.js - PM2 configuration for 8x A100 mining with vLLM
// Each worker spawns its own vLLM server on port 30000 + worker_id
module.exports = {
  apps: [
    // Worker 0 - GPU 0, vLLM auto-spawned on port 30000
    {
      name: 'grail-miner-0',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',  // Use script's shebang (required for Python entry point)
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '0',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '0',
        GRAIL_USE_VLLM: '1',  // vLLM server auto-spawned, URL auto-set
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-0-error.log',
      out_file: '/var/log/grail/worker-0-out.log',
      merge_logs: true,
    },
    // Worker 1 - GPU 1, vLLM auto-spawned on port 30001
    {
      name: 'grail-miner-1',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '1',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '1',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-1-error.log',
      out_file: '/var/log/grail/worker-1-out.log',
      merge_logs: true,
    },
    // Worker 2 - GPU 2, vLLM auto-spawned on port 30002
    {
      name: 'grail-miner-2',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '2',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '2',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-2-error.log',
      out_file: '/var/log/grail/worker-2-out.log',
      merge_logs: true,
    },
    // Worker 3 - GPU 3, vLLM auto-spawned on port 30003
    {
      name: 'grail-miner-3',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '3',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '3',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-3-error.log',
      out_file: '/var/log/grail/worker-3-out.log',
      merge_logs: true,
    },
    // Worker 4 - GPU 4, vLLM auto-spawned on port 30004
    {
      name: 'grail-miner-4',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '4',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '4',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-4-error.log',
      out_file: '/var/log/grail/worker-4-out.log',
      merge_logs: true,
    },
    // Worker 5 - GPU 5, vLLM auto-spawned on port 30005
    {
      name: 'grail-miner-5',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '5',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '5',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-5-error.log',
      out_file: '/var/log/grail/worker-5-out.log',
      merge_logs: true,
    },
    // Worker 6 - GPU 6, vLLM auto-spawned on port 30006
    {
      name: 'grail-miner-6',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '6',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '6',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-6-error.log',
      out_file: '/var/log/grail/worker-6-out.log',
      merge_logs: true,
    },
    // Worker 7 - GPU 7, vLLM auto-spawned on port 30007
    {
      name: 'grail-miner-7',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '7',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '7',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-7-error.log',
      out_file: '/var/log/grail/worker-7-out.log',
      merge_logs: true,
    },
  ],
};
```

> **Note:** The `interpreter: 'none'` setting is required because `.venv/bin/grail` is a Python entry point script with a shebang. This tells PM2 to execute the script directly rather than trying to run it with Node.js.
>
> If not using vLLM, change `GRAIL_USE_VLLM: '0'` and `GRAIL_USE_FLASH_ATTENTION: '1'`.

---

## vLLM Server Management (Optional)

With `GRAIL_USE_VLLM=1`, each miner automatically spawns its own vLLM server on port `30000 + worker_id`. **No manual server management is needed.**

The scripts below are provided for debugging or running vLLM servers independently.

### vLLM Server Architecture

```
GPU 0 -> vLLM :30000 <- grail-miner-0 (auto-spawned)
GPU 1 -> vLLM :30001 <- grail-miner-1 (auto-spawned)
GPU 2 -> vLLM :30002 <- grail-miner-2 (auto-spawned)
...
GPU 7 -> vLLM :30007 <- grail-miner-7 (auto-spawned)
```

### Manual vLLM Server Start (for debugging)

```bash
# Start 8 vLLM servers manually (one per GPU, ports 30000-30007)
bash scripts/start_vllm_servers.sh

# This takes 2-5 minutes to load the model on each GPU
# Script waits until all servers are ready before exiting
```

### Check vLLM Server Status

```bash
bash scripts/check_vllm_servers.sh
```

Expected output:
```
========================================
vLLM Server Status
========================================

GPU 0 (port 30000): HTTP=OK     PID 12345
GPU 1 (port 30001): HTTP=OK     PID 12346
...
GPU 7 (port 30007): HTTP=OK     PID 12352

All servers are ready!

GPU Memory Usage:
  GPU 0: 15234 / 81920 MiB
  ...
```

### Stop vLLM Servers

```bash
bash scripts/stop_vllm_servers.sh
```

### vLLM Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAIL_VLLM_BASE_PORT` | `30000` | Starting port number |
| `GRAIL_VLLM_NUM_SERVERS` | `8` | Number of servers to start |
| `GRAIL_VLLM_GPU_MEMORY_UTIL` | `0.85` | GPU memory fraction for KV cache |
| `GRAIL_VLLM_MAX_MODEL_LEN` | `4096` | Max context length |
| `GRAIL_VLLM_MAX_NUM_SEQS` | `32` | Max concurrent sequences |

### vLLM Troubleshooting

**Servers crash on startup (torch.compile cache error):**
```bash
# Clear corrupted cache and restart
rm -rf /root/.cache/vllm/torch_compile_cache*
bash scripts/start_vllm_servers.sh
```

**Check server logs:**
```bash
cat /var/log/grail/vllm/vllm-server-0.log
```

---

## Running the Miners

### Start All Workers

```bash
# Create directories (running as root - no sudo needed)
mkdir -p /var/log/grail
mkdir -p /var/cache/grail   # Use fast NVMe path if available, e.g., /nvme/grail-cache

# Start all 8 workers
cd /root/Grail
pm2 start ecosystem.config.js

# Save PM2 configuration (auto-restart on reboot)
pm2 save
pm2 startup
```

> **Note on directories:**
> - **Logs**: Can be any writable path. `/var/log/grail` is conventional.
> - **Cache**: Stores model checkpoints (~10-20GB). Put on **fast NVMe** for quick loading.
>   Update `GRAIL_CACHE_DIR` in `.env` if using a different path.

### Management Commands

```bash
# View all worker status
pm2 status

# View logs for all workers
pm2 logs

# View logs for specific worker
pm2 logs grail-miner-0

# Monitor resources (CPU, memory, GPU)
pm2 monit

# Restart all workers
pm2 restart all

# Restart specific worker
pm2 restart grail-miner-0

# Stop all workers
pm2 stop all

# Delete all workers (removes from PM2)
pm2 delete all
```

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Compact GPU summary
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

---

## Performance Tuning

### Recommended Settings by Model Size

| Model Size | Workers | GPUs/Worker | GRAIL_MULTI_GPU | GRAIL_GENERATION_BATCH_SIZE |
|------------|---------|-------------|-----------------|----------------------------|
| 4B         | 8       | 1           | 0               | 8                          |
| 7-8B       | 8       | 1           | 0               | 4                          |
| 14B        | 4       | 2           | 1               | 4                          |
| 30B        | 2       | 4           | 1               | 2                          |
| 70B        | 1       | 8           | 1               | 1                          |

### For 30B Model (4 workers, 2 GPUs each)

Update `ecosystem.config.js`:

```javascript
// Worker 0 - GPUs 0,1
{
  name: 'grail-miner-0',
  script: '.venv/bin/grail',
  args: 'mine',
  interpreter: 'none',
  cwd: '/root/Grail',
  env: {
    GRAIL_WORKER_ID: '0',
    GRAIL_TOTAL_WORKERS: '4',
    CUDA_VISIBLE_DEVICES: '0,1',
    GRAIL_MULTI_GPU: '1',
    GRAIL_USE_VLLM: '1',
    GRAIL_USE_FLASH_ATTENTION: '0',
    GRAIL_GENERATION_BATCH_SIZE: '4',
  },
  // ...
},
// Worker 1 - GPUs 2,3
{
  interpreter: 'none',
  env: {
    GRAIL_WORKER_ID: '1',
    GRAIL_TOTAL_WORKERS: '4',
    CUDA_VISIBLE_DEVICES: '2,3',
    GRAIL_MULTI_GPU: '1',
    GRAIL_USE_VLLM: '1',
    GRAIL_USE_FLASH_ATTENTION: '0',
    // ...
  },
},
// Worker 2 - GPUs 4,5
// Worker 3 - GPUs 6,7
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
GRAIL_GENERATION_BATCH_SIZE=4  # or 2

# Or enable quantization for larger models
GRAIL_QUANTIZATION=int8
```

**2. Workers Competing for Checkpoint Download**
```
# This is handled automatically - workers coordinate via file locks
# Only one worker downloads, others wait
```

**3. Slow Upload Speed**
```bash
# Adjust bandwidth estimate
GRAIL_UPLOAD_BANDWIDTH_MBPS=500  # Tune based on actual speed

# Test upload speed
curl -o /dev/null -w "%{speed_upload}\n" --data-binary @/path/to/testfile https://httpbin.org/post
```

**4. Flash Attention Not Working**
```bash
# Reinstall - builds from source (takes 10-15 min)
uv pip uninstall flash-attn
uv pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print(flash_attn.__version__)"

# If it still fails, skip it - mining works without flash-attn (just slower)
```

### Health Check Script

Create `check_miners.sh`:

```bash
#!/bin/bash
echo "=== GRAIL Miner Status ==="
echo ""

# PM2 Status
echo "PM2 Workers:"
pm2 jlist | python3 -c "
import sys, json
data = json.load(sys.stdin)
for app in data:
    name = app['name']
    status = app['pm2_env']['status']
    restarts = app['pm2_env']['restart_time']
    uptime = app['pm2_env'].get('pm_uptime', 0)
    print(f'  {name}: {status} (restarts: {restarts})')
"
echo ""

# GPU Usage
echo "GPU Usage:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | while read line; do
    echo "  GPU $line"
done
echo ""

# Recent Logs (errors only)
echo "Recent Errors (last 10):"
grep -h "ERROR\|Exception\|Traceback" /var/log/grail/worker-*-error.log 2>/dev/null | tail -10
```

---

## Expected Performance

With 8x A100 80GB running the 4B model:

| Metric | Expected Value |
|--------|---------------|
| Rollouts per worker per window | ~250-400 |
| Total rollouts per window (8 workers) | ~2000-3200 |
| GPU utilization | 70-90% |
| VRAM usage per GPU | ~20-30GB |
| Network upload per window | ~50-200MB |

### Scoring Impact

With the superlinear scoring formula `score = unique_rollouts^4.0`:
- 1000 rollouts: score = 1,000,000,000,000 (10^12)
- 2000 rollouts: score = 16,000,000,000,000 (16x more!)
- 3000 rollouts: score = 81,000,000,000,000 (81x more!)

**More workers = exponentially higher score.**

---

## Quick Start Summary

```bash
# 1. Install uv and clone repo
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/one-covenant/grail && cd grail

# 2. Install dependencies
uv venv --python 3.10 && source .venv/bin/activate
uv sync
uv pip install bittensor-cli                     # Wallet management
uv pip install orjson

# 2b. Choose ONE performance option:
# Option A: vLLM (recommended, 5-10x faster, easier setup)
bash scripts/setup_vllm_env.sh
# Option B: Flash Attention (1.3x faster, slow build)
# uv pip install flash-attn --no-build-isolation   # Takes 10-15 min

# 3. Setup wallet (if new server)
btcli w regen_coldkey --wallet-name YOUR_WALLET --wallet_path "~/.bittensor/wallets/" --mnemonic "your 12 words" --overwrite
btcli w regen_hotkey --wallet-name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY --wallet_path "~/.bittensor/wallets/" --mnemonic "your 12 words" --overwrite
btcli w list  # Verify wallet

# 4. Configure environment
cp .env.example .env
nano .env  # Edit with your wallet names, R2 credentials, etc.

# 5. Create ecosystem.config.js (copy from guide above, update paths)

# 6. Create directories and start miners (vLLM auto-spawned)
mkdir -p /var/log/grail /var/cache/grail
pm2 start ecosystem.config.js
pm2 save

# 7. Monitor
pm2 logs
watch -n 1 nvidia-smi
```
