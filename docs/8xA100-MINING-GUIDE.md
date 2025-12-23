# GRAIL Mining Guide: 8x A100 80GB Setup

This guide covers running the GRAIL miner on a server with 8x NVIDIA A100 80GB GPUs using PM2 for process management.

## Architecture Overview

For maximum throughput on 8x A100 with a **4B model** (current default), we run **8 parallel workers**, each using 1 GPU. This provides ~8x throughput compared to single-GPU mining.

```
┌─────────────────────────────────────────────────────────────────┐
│                     8x A100 Server                              │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┤
│ Worker 0│ Worker 1│ Worker 2│ Worker 3│ Worker 4│ Worker 5│ ... │
│ (LEADER)│  GPU 1  │  GPU 2  │  GPU 3  │  GPU 4  │  GPU 5  │ ... │
│  GPU 0  │         │         │         │         │         │     │
├─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────┤
│              Shared: Checkpoint Cache, R2 Storage               │
└─────────────────────────────────────────────────────────────────┘
```

### Leader-Follower Pattern

To avoid redundant blockchain operations, workers use a **leader-follower pattern**:

- **Worker 0 (Leader)**: Does all blockchain initialization:
  - Connects to subtensor and fetches metagraph
  - Initializes chain manager (spawns background process)
  - Commits R2 credentials to blockchain
  - Writes a barrier file when ready

- **Workers 1-7 (Followers)**: Wait for leader, then proceed directly to mining:
  - Wait for leader's barrier file (up to 5 minutes)
  - Read shared data (trainer bucket info) from barrier
  - Skip blockchain initialization entirely
  - Start mining immediately

**Benefits:**
- 8 chain_worker processes → 1 process (saves memory)
- 8 blockchain transactions → 1 transaction (faster startup)
- Workers 1-7 start mining faster (no blockchain wait)

The barrier file is stored at `~/.cache/grail/.worker-barrier/leader-ready.json`.

## Prerequisites

### 1. Install System Dependencies

```bash
# CUDA drivers (should be pre-installed on cloud instances)
nvidia-smi  # Verify GPUs are visible

# Tested configuration (2448+ rollouts achieved):
#   Driver: 570.172.08
#   CUDA: 12.8
#   GPUs: 8x NVIDIA A100-SXM4-80GB

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
# IMPORTANT: Bucket name MUST equal account_id for validators to find your files!
# The chain commitment format uses account_id as the bucket identifier.
R2_ACCOUNT_ID=your_account_id
R2_BUCKET_NAME=your_account_id  # Must match R2_ACCOUNT_ID!
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
# IMPORTANT: Do NOT set these in .env - they are set per-worker via PM2!
# If set here, they will be ignored (PM2 env vars take precedence).
# GRAIL_WORKER_ID=0        # Set in ecosystem.config.js per worker
# GRAIL_TOTAL_WORKERS=8    # Set in ecosystem.config.js per worker
# CUDA_VISIBLE_DEVICES=0   # Set in ecosystem.config.js per worker

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
// ecosystem.config.js - PM2 configuration for 8x A100 mining with vLLM backend
// Each worker runs on its own GPU with vLLM (auto-spawned server)
//
// LEADER-FOLLOWER PATTERN:
//   Worker 0 (leader) initializes blockchain/checkpoints and signals ready via barrier file.
//   Workers 1-7 (followers) wait for leader's barrier signal before mining (no PM2 delay needed).
//
// SETUP:
//   1. pm2 start ecosystem.config.js      # Start miners with vLLM backend
//
// STOP:
//   pm2 stop all

module.exports = {
  apps: [
    // Worker 0 (LEADER) - GPU 0, starts immediately
    {
      name: 'grail-miner-0',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '0',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '0',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',  // Aggressive: 1 block = ~12s buffer
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-0-error.log',
      out_file: '/var/log/grail/worker-0-out.log',
      merge_logs: true,
    },
    // Worker 1 (FOLLOWER) - GPU 1, waits for leader barrier
    {
      name: 'grail-miner-1',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '1',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '1',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-1-error.log',
      out_file: '/var/log/grail/worker-1-out.log',
      merge_logs: true,
    },
    // Worker 2 (FOLLOWER) - GPU 2, waits for leader barrier
    {
      name: 'grail-miner-2',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '2',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '2',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-2-error.log',
      out_file: '/var/log/grail/worker-2-out.log',
      merge_logs: true,
    },
    // Worker 3 (FOLLOWER) - GPU 3, waits for leader barrier
    {
      name: 'grail-miner-3',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '3',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '3',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-3-error.log',
      out_file: '/var/log/grail/worker-3-out.log',
      merge_logs: true,
    },
    // Worker 4 (FOLLOWER) - GPU 4, waits for leader barrier
    {
      name: 'grail-miner-4',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '4',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '4',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-4-error.log',
      out_file: '/var/log/grail/worker-4-out.log',
      merge_logs: true,
    },
    // Worker 5 (FOLLOWER) - GPU 5, waits for leader barrier
    {
      name: 'grail-miner-5',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '5',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '5',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-5-error.log',
      out_file: '/var/log/grail/worker-5-out.log',
      merge_logs: true,
    },
    // Worker 6 (FOLLOWER) - GPU 6, waits for leader barrier
    {
      name: 'grail-miner-6',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '6',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '6',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-6-error.log',
      out_file: '/var/log/grail/worker-6-out.log',
      merge_logs: true,
    },
    // Worker 7 (FOLLOWER) - GPU 7, waits for leader barrier
    {
      name: 'grail-miner-7',
      script: '.venv/bin/grail',
      args: '-vv mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '7',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '7',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_MINER_SAFETY_BLOCKS: '1',
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
> **Barrier-based Synchronization:** Unlike older configurations that used `sleep` delays, all workers now start immediately. The leader-follower barrier mechanism handles synchronization automatically - followers wait for the leader's barrier file before proceeding to mine.
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

### Summary Log Monitoring

Each worker logs a `[SUMMARY]` line after completing a window, making it easy to monitor progress:

```bash
# Live filtered view of summary lines from all workers
tail -f /var/log/grail/worker-*-out.log | grep "\[SUMMARY\]"

# Filter summary lines from existing logs
grep "\[SUMMARY\]" /var/log/grail/worker-*-out.log
```

**Example output:**
```
[SUMMARY] W0 | window=7149180 | rollouts=156 | UPLOADED | 22:45:30
[SUMMARY] W1 | window=7149180 | rollouts=18 | STAGED | 22:45:28
[SUMMARY] W2 | window=7149180 | rollouts=20 | STAGED | 22:45:27
[SUMMARY] W3 | window=7149180 | rollouts=19 | STAGED | 22:45:26
...
```

- **W0 UPLOADED**: Leader aggregated all rollouts and uploaded to R2
- **W1-W7 STAGED**: Followers staged their rollouts for leader to aggregate

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Compact GPU summary
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

---

## Performance Tuning

### Safety Blocks Tuning

The `GRAIL_MINER_SAFETY_BLOCKS` environment variable controls how many blocks before the deadline miners stop generating new rollouts. This is a critical tuning parameter:

| Setting | Buffer Time | Use Case |
|---------|-------------|----------|
| 1 | ~12 seconds | **Aggressive (vLLM recommended)** - Maximum throughput |
| 2 | ~24 seconds | Conservative - For slower backends or unstable networks |
| 3+ | ~36+ seconds | Very conservative - Only if experiencing missed deadlines |

**How it works:**

1. The miner calculates `blocks_needed_for_next_gen` which estimates how many blocks are needed to complete the next generation batch plus upload time
2. When `blocks_remaining <= blocks_needed_for_next_gen`, the miner stops generating and uploads
3. The safety blocks value is added as buffer to this calculation

**Recommended settings:**

- **vLLM backend**: Set to `1` - vLLM is fast enough that you don't need extra buffer. This was tested to achieve 2400+ rollouts per window.
- **HuggingFace backend**: Set to `2` or `3` - Slower inference means more buffer is safer
- **Flash Attention backend**: Set to `2` - Middle ground

**Symptoms of incorrect settings:**

- **Safety blocks too high**: Uploading 60+ seconds before deadline (wasted mining time)
- **Safety blocks too low**: Missing deadlines, incomplete uploads, or "duplicate nonce" errors

**Monitoring upload timing:**

Check your validator logs for upload timing:
```
Parquet file ... uploaded 15s before deadline  # Good - close to deadline
Parquet file ... uploaded 69s before deadline  # Too early - reduce safety blocks
```

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

**1. All Workers Think They're Worker 0 (Leader)**

If you see multiple workers logging "Leader signaled ready" or all workers trying to initialize blockchain:

```bash
# Check logs for worker ID detection
pm2 logs | grep "WorkerConfig"
# Should show: WorkerConfig: GRAIL_WORKER_ID='4', GRAIL_TOTAL_WORKERS='8' (parsed: 4/8)
```

**Cause:** The `.env` file contains `GRAIL_WORKER_ID=0` which overrides PM2's environment variables.

**Fix:** Remove `GRAIL_WORKER_ID` and `GRAIL_TOTAL_WORKERS` from your `.env` file. These should ONLY be set via PM2's `ecosystem.config.js`.

```bash
# Edit .env and remove these lines (if present):
# GRAIL_WORKER_ID=0
# GRAIL_TOTAL_WORKERS=1

# Then restart PM2 completely:
pm2 delete all && pm2 start ecosystem.config.js
```

> **Technical note:** The grail package uses `load_dotenv(override=False)` so PM2 env vars take precedence. But if you have old code with `override=True`, update to latest.

**2. CUDA Out of Memory**
```bash
# Reduce batch size
GRAIL_GENERATION_BATCH_SIZE=4  # or 2

# Or enable quantization for larger models
GRAIL_QUANTIZATION=int8
```

**3. Workers Stuck Waiting for Checkpoint Download**

If workers log "waiting for checkpoint (another worker downloading)" indefinitely:

```bash
# Check for stale lock files
ls -la ~/.cache/grail/checkpoints/.locks/

# If you see a .lock file without a matching .complete file, remove it
rm ~/.cache/grail/checkpoints/.locks/checkpoint-*.lock
pm2 restart all
```

> **Note:** As of the latest version, stale locks (>60s old) are automatically cleaned up on worker startup. If you still see this issue, update to the latest code.

**4. Slow Upload Speed**
```bash
# Adjust bandwidth estimate
GRAIL_UPLOAD_BANDWIDTH_MBPS=500  # Tune based on actual speed

# Test upload speed
curl -o /dev/null -w "%{speed_upload}\n" --data-binary @/path/to/testfile https://httpbin.org/post
```

**5. Flash Attention Not Working**
```bash
# Reinstall - builds from source (takes 10-15 min)
uv pip uninstall flash-attn
uv pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print(flash_attn.__version__)"

# If it still fails, skip it - mining works without flash-attn (just slower)
```

**6. Duplicate Nonce Error After PM2 Restart**

If you see validation failures with "Duplicate nonce" after restarting PM2:

```
Duplicate nonce 45; invalidating uid
```

**Cause:** Stale staging files from the previous run mixed with new rollouts. When PM2 restarts, old `.json` files in the staging directory can be picked up alongside new ones, causing the same problem index (nonce) to appear twice.

**Automatic Fix:** As of the latest version, the leader worker automatically cleans up all stale staging files on startup. Update to the latest code to get this fix.

**Manual Fix (if needed):**
```bash
# Stop all miners
pm2 stop all

# Clear staging directory
rm -rf ~/.cache/grail/.worker-barrier/*.json
rm -rf ~/.cache/grail/.worker-barrier/*.done
rm -rf ~/.cache/grail/.worker-barrier/*.tmp

# Restart miners
pm2 start all
```

**Prevention:** The barrier system now includes automatic cleanup on leader startup, so this should not occur with current code.

**7. Gap Truncation (Rollouts Discarded)**

If you see logs like:
```
⚠️ GAP at problem 129: Truncated 320 rollouts
```

**Cause:** With round-robin problem distribution (worker 0 handles problems 0,8,16...; worker 1 handles 1,9,17...; etc.), if one worker doesn't complete its assigned problem before the window ends, there's a gap in the problem indices.

**Why it matters:** The validator requires contiguous problem indices. A gap at problem 129 means all rollouts for problems 130+ would fail validation anyway, so they're discarded.

**Solutions:**
- Ensure all workers have similar generation speeds (same GPU types)
- Reduce `GRAIL_MINER_SAFETY_BLOCKS` so workers stop at more similar times
- With well-tuned settings, gaps should affect <5% of windows

**8. Duplicate Nonce Error with Batch Size > 10 (Formula Collision)**

If you see validation failures with "Duplicate nonce" even with fresh staging files:

```
Duplicate nonce 10; invalidating uid
```

**Cause:** In older code versions, the nonce formula used a multiplier of 10:
```python
rollout_nonce = base_nonce * 10 + rollout_idx
```

With `GRAIL_GENERATION_BATCH_SIZE=16`, rollout indices go 0-15. This causes collisions:
- Rollout group 0, index 10 → nonce = 0×10 + 10 = **10**
- Rollout group 1, index 0 → nonce = 1×10 + 0 = **10** (collision!)

**Fix:** Update to the latest code. The formula now uses a multiplier of 100:
```python
rollout_nonce = base_nonce * 100 + rollout_idx
```

This supports batch sizes up to 99 without collisions.

**Verification:** Use the validation script to check for duplicate nonces:
```bash
python scripts/validate_parquet.py /path/to/your/rollouts.parquet
```

The script will show:
- `Duplicate nonces detected` if the bug is present
- Smart diagnosis to identify formula collision vs stale file issues
- 50-entry sample dump for manual inspection

**Key indicator:** If duplicates follow a pattern like nonces 10-15 appearing twice (once from high rollout indices, once from low indices in next group), it's the formula collision bug.

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

## Rollout Aggregation

In multi-worker setups, all workers generate rollouts but only **worker 0 (leader) uploads the aggregated file** to R2. This ensures the validator receives a single file with all rollouts.

### How It Works

1. Each worker generates rollouts and stages them locally
2. Worker 0 waits for all workers to complete staging
3. Worker 0 aggregates all rollouts into a single file
4. Worker 0 uploads the combined file to R2

### Verifying Aggregation

Check the logs to verify aggregation is working:

```bash
# See all staging activity
pm2 logs | grep -E "(staged|aggregated|Leader:)"

# Check worker 0's logs for aggregation
tail -f /var/log/grail/worker-0-out.log | grep -E "(aggregated|staged)"
```

**Expected log output:**
```
Worker 0 staged 175 rollouts for window 7149180
Worker 1 staged 172 rollouts for window 7149180
Worker 2 staged 168 rollouts for window 7149180
...
Worker 7 staged 180 rollouts for window 7149180
Leader: 7/7 workers completed staging for window 7149180 (waited 2.5s)
Leader aggregated 1392 total rollouts from 8 workers for window 7149180
Successfully uploaded window 7149180 with 1392 aggregated rollouts
```

The total rollout count (1392 in this example) should be approximately 8x what each individual worker staged.

### First Window Performance

The **first window takes longer** due to:
- Checkpoint download (5-10GB model files)
- vLLM warmup and CUDA kernel compilation
- Leader blockchain initialization

Subsequent windows are faster as checkpoints are cached and vLLM is warmed up.

---

## Seed Derivation and Validation

Understanding how the validator verifies rollouts is critical for multi-GPU mining. This section explains why rollout ordering matters.

### How Problems Are Generated

Each problem is generated deterministically from a **seed** derived from three inputs:

```
seed = derive_env_seed(miner_hotkey, window_block_hash, problem_index)
```

- **miner_hotkey**: Your wallet's SS58 address (unique to you)
- **window_block_hash**: Randomness from the blockchain for this window (same for all miners)
- **problem_index**: Which problem (0, 1, 2, 3...)

This determinism allows the validator to **reconstruct the exact same problem** independently.

### How Validation Works

When the validator checks your rollouts:

1. **Reads rollouts from your uploaded file sequentially**
2. **Assigns `group_index` based on file position** (0, 1, 2, 3...)
3. **Derives the seed**: `seed = derive_env_seed(your_hotkey, window_hash, group_index)`
4. **Reconstructs the expected prompt** from that seed
5. **Compares** against the actual prompt in your rollout

If the prompts match → `env_prompt_valid = True` ✓
If they don't match → `env_prompt_valid = False` ✗

### Why File Order Matters (Multi-GPU)

With multiple workers, problems are distributed round-robin:
- Worker 0 handles: 0, 8, 16, 24, 32...
- Worker 1 handles: 1, 9, 17, 25, 33...
- Worker 2 handles: 2, 10, 18, 26, 34...
- etc.

**Without sorting**, rollouts would be uploaded in worker order:
```
File position 0 → problem 0 (worker 0)  ✓ validator uses group_index=0
File position 1 → problem 8 (worker 0)  ✗ validator uses group_index=1, but seed was for problem 8!
File position 2 → problem 16 (worker 0) ✗ validator uses group_index=2, but seed was for problem 16!
...
```

**With sorting by problem_index**, rollouts are in correct order:
```
File position 0 → problem 0  ✓ validator uses group_index=0
File position 1 → problem 1  ✓ validator uses group_index=1
File position 2 → problem 2  ✓ validator uses group_index=2
...
```

### Gap Handling

If a worker doesn't finish a problem before the window ends, there's a **gap** in the problem indices:

```
Problems completed: 0, 1, 2, 3, 4, 6, 7, 8...  (missing 5!)
```

Without gap handling:
```
File position 5 → problem 6  ✗ validator expects problem 5's prompt!
File position 6 → problem 7  ✗ validator expects problem 6's prompt!
...all subsequent rollouts fail
```

The miner automatically **truncates at the first gap**, keeping only contiguous problems:
```
File positions 0-4 → problems 0-4 (all pass validation)
Problems 6+ are dropped (would fail anyway)
```

### What You'll See in Logs

**Successful aggregation (no gaps):**
```
✅ No gaps: 72 contiguous problems [0-71]
Leader aggregated 1152 total rollouts from 8 workers for window 7155540 (sorted by problem_index)
```

**Aggregation with gap detected:**
```
⚠️ GAP at problem 59: Truncated 160 rollouts (keeping 59 contiguous problems [0-58])
Leader aggregated 944 total rollouts from 8 workers for window 7155540 (sorted by problem_index)
```

The truncated rollouts would have failed validation anyway, so it's better to upload fewer valid rollouts than more invalid ones.

### Minimizing Gaps

Gaps occur when one worker finishes fewer problems than others. To minimize:

1. **Use similar GPUs**: Mixed GPU types (e.g., A100 + H100) cause timing differences
2. **Increase safety margin**: Set `GRAIL_MINER_SAFETY_BLOCKS=4` (default 3) to stop generation earlier
3. **Balance workload**: Ensure all workers have similar generation speeds

With well-balanced workers, gaps should be rare (< 5% of windows).

### Local Parquet Saving

To save parquet files locally for debugging and validation, enable local saving:

```bash
# Add to .env or set in ecosystem.config.js
GRAIL_SAVE_PARQUET_LOCALLY=1
```

Files are saved to:
```
~/.cache/grail/uploads/{hotkey}-window-{window}.parquet
# Or if GRAIL_CACHE_DIR is set:
$GRAIL_CACHE_DIR/uploads/{hotkey}-window-{window}.parquet
```

### Downloading Parquet Files from R2

To download parquet files from R2 for debugging or validation:

```bash
# First install boto3 if not already installed
uv pip install boto3

# List all parquet files in your bucket
python scripts/download_parquet.py --list

# List files for a specific hotkey
python scripts/download_parquet.py --list --hotkey 5Gxxx...

# Download specific file by hotkey and window
python scripts/download_parquet.py --hotkey 5Gxxx... --window 7155690

# Download to specific directory
python scripts/download_parquet.py --hotkey 5Gxxx... --window 7155690 --output /tmp/

# Download latest N files for a hotkey
python scripts/download_parquet.py --hotkey 5Gxxx... --latest 5
```

This uses your existing R2 credentials from `.env` (R2_ACCOUNT_ID, R2_BUCKET_ID, R2_READ_ACCESS_KEY_ID, R2_READ_SECRET_ACCESS_KEY).

### Self-Validation Tool

You can validate your parquet files locally before the validator checks them:

```bash
# Validate a local parquet file
python scripts/validate_parquet.py /path/to/hotkey-window-123456.parquet

# Verbose output (show each rollout)
python scripts/validate_parquet.py /path/to/file.parquet -v

# Show more failure details
python scripts/validate_parquet.py /path/to/file.parquet --show-failures 10
```

**Example output (all valid):**
```
Loading parquet file: 5Gxxx-window-7155540.parquet
  Hotkey: 5Gxxx...
  Window: 7155540
  Rollouts: 944

Fetching window hash from chain...
  Window hash: 0xabc123def456...

Analyzing rollout ordering...
  Rollouts are sorted by rollout_group
  Group range: 0 to 58
  No gaps: contiguous from 0 to 58

Validating rollouts...

============================================================
VALIDATION SUMMARY
============================================================
Valid rollouts:   944/944 (100.0%)
Failed rollouts:  0

✅ All rollouts should pass env_prompt_valid!
```

**Example output (failures detected):**
```
Analyzing rollout ordering...
  WARNING: Rollouts are NOT sorted by rollout_group!
  First 10 groups in file order: [0, 8, 16, 24, 1, 9, 17, 25, 2, 10]

Validating rollouts...
  [FAIL] Token mismatch at position 0: expected=128000, actual=128001 (file_pos=1, rollout_group=8, seed=...)

============================================================
VALIDATION SUMMARY
============================================================
Valid rollouts:   128/1024 (12.5%)
Failed rollouts:  896

❌ 896 rollouts will FAIL env_prompt_valid
```

This script uses the **exact same seed derivation** as the validator, so if it passes locally, it should pass validation on-chain.

---

## Expected Performance

With 8x A100 80GB running the 4B model:

| Metric | HuggingFace Backend | vLLM Backend |
|--------|---------------------|--------------|
| Rollouts per worker per window | ~150-250 | ~300-400 |
| Total rollouts per window (8 workers) | ~1200-2000 | ~2400-3200 |
| GPU utilization | 70-90% | 80-95% |
| VRAM usage per GPU | ~20-30GB | ~15-25GB |
| Network upload per window | ~50-200MB | ~100-300MB |

**Real-world results with optimized vLLM setup:**
- 2448 rollouts achieved with 8x A100, `GRAIL_MINER_SAFETY_BLOCKS=1`, batch size 16
- This represents ~306 rollouts per worker per window

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
