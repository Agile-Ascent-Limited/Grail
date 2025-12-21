# GRAIL Throughput Benchmark Guide

Test your mining throughput **without connecting to Bittensor network**. This helps tune your setup before going live.

## Prerequisites

- Configured bittensor wallet (for GRAIL proof generation)
- Set wallet environment variables:
  ```bash
  export BT_WALLET_COLD=your_coldkey_name
  export BT_WALLET_HOT=your_hotkey_name
  ```

## Quick Start

```bash
cd /path/to/grail
source .venv/bin/activate

# Basic benchmark (single GPU, default model)
python scripts/benchmark_throughput.py

# Test with 8 workers projection (for 8x A100)
python scripts/benchmark_throughput.py --workers 8 --num-problems 20

# Recommended: 8x A100 with Flash Attention (full test)
GRAIL_USE_FLASH_ATTENTION=1 python scripts/benchmark_throughput.py --num-problems 10 --workers 8
```

## What It Measures

- **Rollouts per second**: Raw generation throughput
- **Problems per second**: How many SAT problems processed
- **GPU memory usage**: Peak VRAM consumption
- **Success rate**: Percentage of rollouts that solve the problem

Requires wallet for GRAIL proofs. No R2 credentials or network connection required.

---

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Model to benchmark |
| `--batch-sizes` | `4` | Comma-separated batch sizes (e.g., `1,2,4,8`) |
| `--num-problems` | `10` | Problems per benchmark run |
| `--workers` | `1` | Project throughput for N workers |
| `--full-benchmark` | - | Test all batch sizes 1-16 |
| `--seed` | `42` | Random seed for reproducibility |

---

## Example Commands

### 1. Find Optimal Batch Size

Test different batch sizes to find the best throughput for your GPU:

```bash
python scripts/benchmark_throughput.py --batch-sizes 1,2,4,8,16 --num-problems 10
```

### 2. Project 8-Worker Throughput

See what 8 parallel workers would produce:

```bash
python scripts/benchmark_throughput.py --workers 8 --batch-sizes 8 --num-problems 20
```

### 3. Test with Flash Attention

```bash
GRAIL_USE_FLASH_ATTENTION=1 python scripts/benchmark_throughput.py --num-problems 20
```

### 4. Test Multi-GPU (Single Worker Using All GPUs)

```bash
GRAIL_MULTI_GPU=1 python scripts/benchmark_throughput.py --num-problems 10
```

### 5. Full Benchmark Suite

Comprehensive test of all configurations:

```bash
python scripts/benchmark_throughput.py --full-benchmark --num-problems 15
```

### 6. Test Specific Checkpoint

```bash
python scripts/benchmark_throughput.py --model /var/cache/grail/checkpoint-1234
```

---

## Understanding Results

```
================================================================================
BENCHMARK RESULTS
================================================================================

Batch    Problems   Rollouts   Time(s)    Roll/s     Prob/s     GPU(GB)    Success%
--------------------------------------------------------------------------------
4        10         160        45.2       3.5        0.22       24.50      65.0
8        10         160        38.1       4.2        0.26       32.10      63.0

--------------------------------------------------------------------------------
PROJECTED 8-WORKER THROUGHPUT:
--------------------------------------------------------------------------------
  Batch 4: 28.0 rollouts/s (~5040 rollouts per 30-block window)
  Batch 8: 33.6 rollouts/s (~6048 rollouts per 30-block window)

================================================================================
BEST: batch_size=8 -> 4.2 rollouts/s
       With 8 workers: ~33.6 rollouts/s
================================================================================
```

### Key Metrics

| Metric | What It Means |
|--------|---------------|
| **Roll/s** | Rollouts generated per second (higher = better) |
| **Prob/s** | SAT problems completed per second |
| **GPU(GB)** | Peak VRAM usage - ensure headroom below 80GB |
| **Success%** | Rollouts that solved the problem (affects scoring) |

### Window Throughput

A scoring window is 50 blocks (~10 minutes). The projection shows expected rollouts per window.

With the superlinear scoring `score = rollouts^4.0`:
- 2000 rollouts = 16x score of 1000 rollouts
- 3000 rollouts = 81x score of 1000 rollouts

---

## Recommended Settings by GPU

### Single A100 80GB

```bash
# Optimal for 4B model
GRAIL_USE_FLASH_ATTENTION=1 \
python scripts/benchmark_throughput.py --batch-sizes 8 --num-problems 20
```

Expected: ~3-5 rollouts/s

### 8x A100 80GB (8 Workers)

```bash
# Test single worker performance, project to 8
GRAIL_USE_FLASH_ATTENTION=1 \
python scripts/benchmark_throughput.py --batch-sizes 8 --workers 8 --num-problems 20
```

Expected: ~25-40 rollouts/s total (8 workers)

### Consumer GPU (RTX 3090/4090)

```bash
# Lower batch size for 24GB VRAM
python scripts/benchmark_throughput.py --batch-sizes 1,2,4 --num-problems 10
```

Expected: ~1-2 rollouts/s

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python scripts/benchmark_throughput.py --batch-sizes 1,2
```

### Slow Performance

1. Enable Flash Attention:
   ```bash
   uv pip install flash-attn --no-build-isolation  # Takes 10-15 min to build
   GRAIL_USE_FLASH_ATTENTION=1 python scripts/benchmark_throughput.py
   ```

2. Install orjson:
   ```bash
   uv pip install orjson
   ```

### Model Download Slow

Pre-download the model:
```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

### Warmup Fails

The warmup failure warning is normal for benchmark-only mode (no wallet). The actual benchmark will still run.

---

## Interpreting for Production

Once you have benchmark results, use them to configure your PM2 setup:

1. **Best batch size** → Set `GRAIL_GENERATION_BATCH_SIZE` in ecosystem.config.js
2. **GPU memory** → Ensure 10-15GB headroom for checkpoint loading
3. **Projected throughput** → Compare with leaderboard to estimate competitiveness

### Example: Benchmark Shows 4.2 rollouts/s per Worker

With 8 workers:
- ~33.6 rollouts/s total
- ~6000 rollouts per 10-minute window
- Score: 6000^4 = 1.3 × 10^15

This would be competitive on the network.
