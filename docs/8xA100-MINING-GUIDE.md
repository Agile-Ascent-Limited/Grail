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

# Node.js and PM2
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g pm2
```

### 2. Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

### 3. Install Python Dependencies

Follow the official installation from [docs/miner.md](miner.md):

```bash
cd /path/to/grail

# Create venv with Python 3.10 (required by grail) and install
uv venv --python 3.10 && source .venv/bin/activate
uv sync

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

### 4. Register Your Miner (if not already done)

```bash
# Register on subnet 81
btcli subnet register --netuid 81 --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY
```

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
// ecosystem.config.js - PM2 configuration for 8x A100 mining
module.exports = {
  apps: [
    // Worker 0 - GPU 0
    {
      name: 'grail-miner-0',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '0',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '0',
        GRAIL_USE_FLASH_ATTENTION: '1',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-0-error.log',
      out_file: '/var/log/grail/worker-0-out.log',
      merge_logs: true,
    },
    // Worker 1 - GPU 1
    {
      name: 'grail-miner-1',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '1',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '1',
        GRAIL_USE_FLASH_ATTENTION: '1',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-1-error.log',
      out_file: '/var/log/grail/worker-1-out.log',
      merge_logs: true,
    },
    // Worker 2 - GPU 2
    {
      name: 'grail-miner-2',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '2',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '2',
        GRAIL_USE_FLASH_ATTENTION: '1',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-2-error.log',
      out_file: '/var/log/grail/worker-2-out.log',
      merge_logs: true,
    },
    // Worker 3 - GPU 3
    {
      name: 'grail-miner-3',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '3',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '3',
        GRAIL_USE_FLASH_ATTENTION: '1',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-3-error.log',
      out_file: '/var/log/grail/worker-3-out.log',
      merge_logs: true,
    },
    // Worker 4 - GPU 4
    {
      name: 'grail-miner-4',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '4',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '4',
        GRAIL_USE_FLASH_ATTENTION: '1',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-4-error.log',
      out_file: '/var/log/grail/worker-4-out.log',
      merge_logs: true,
    },
    // Worker 5 - GPU 5
    {
      name: 'grail-miner-5',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '5',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '5',
        GRAIL_USE_FLASH_ATTENTION: '1',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-5-error.log',
      out_file: '/var/log/grail/worker-5-out.log',
      merge_logs: true,
    },
    // Worker 6 - GPU 6
    {
      name: 'grail-miner-6',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '6',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '6',
        GRAIL_USE_FLASH_ATTENTION: '1',
        GRAIL_GENERATION_BATCH_SIZE: '8',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-6-error.log',
      out_file: '/var/log/grail/worker-6-out.log',
      merge_logs: true,
    },
    // Worker 7 - GPU 7
    {
      name: 'grail-miner-7',
      script: '.venv/bin/grail',
      args: 'mine',
      cwd: '/path/to/grail',
      env: {
        GRAIL_WORKER_ID: '7',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '7',
        GRAIL_USE_FLASH_ATTENTION: '1',
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

---

## Running the Miners

### Start All Workers

```bash
# Create directories (running as root - no sudo needed)
mkdir -p /var/log/grail
mkdir -p /var/cache/grail   # Use fast NVMe path if available, e.g., /nvme/grail-cache

# Start all 8 workers
cd /path/to/grail
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
  cwd: '/path/to/grail',
  env: {
    GRAIL_WORKER_ID: '0',
    GRAIL_TOTAL_WORKERS: '4',
    CUDA_VISIBLE_DEVICES: '0,1',
    GRAIL_MULTI_GPU: '1',
    GRAIL_USE_FLASH_ATTENTION: '1',
    GRAIL_GENERATION_BATCH_SIZE: '4',
  },
  // ...
},
// Worker 1 - GPUs 2,3
{
  env: {
    GRAIL_WORKER_ID: '1',
    GRAIL_TOTAL_WORKERS: '4',
    CUDA_VISIBLE_DEVICES: '2,3',
    GRAIL_MULTI_GPU: '1',
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
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# 2. Setup
cd /path/to/grail
uv venv --python 3.10 && source .venv/bin/activate
uv sync
uv pip install orjson
uv pip install flash-attn --no-build-isolation  # Takes 10-15 min, optional

# 3. Configure
cp .env.example .env
nano .env  # Edit with your wallet, R2 credentials, etc.

# 4. Create ecosystem.config.js (copy from above, update paths)

# 5. Start
mkdir -p /var/log/grail /var/cache/grail
pm2 start ecosystem.config.js
pm2 save

# 6. Monitor
pm2 logs
watch -n 1 nvidia-smi
```
