# Local Testing Guide

This guide explains how to test your miner's proofs locally before submitting to the network.

## Overview

Running a validator in `--test-mode` alongside your miner lets you:
- Verify proofs pass before waiting for network validation
- Debug precision issues across GPU architectures
- Test code changes without risking failed submissions

## Prerequisites

- Same `.env` file configured for both miner and validator
- At least 2 GPUs (one for each) or run sequentially
- Checkpoint cache populated

## Quick Start

**Important:** Start the validator first and wait for model downloads to complete before starting miners. This ensures both use the same cached checkpoint.

### Step 1 - Start Validator First (wait for downloads)
```bash
CUDA_VISIBLE_DEVICES=7 grail -vv validate --test-mode
```

Wait until you see:
```
Loading checkpoint for window XXXXX from /root/.cache/grail/checkpoints/checkpoint-XXXXX
```

The large model files (~8GB) will download to `/root/.cache/grail/checkpoints/`. Once complete, miners will reuse this cache.

### Step 2 - Start Miners (after downloads complete)
```bash
pm2 start ecosystem-testing.config.js --only grail-miner-0,grail-miner-1,...
```

Or manually:
```bash
CUDA_VISIBLE_DEVICES=0 grail mine
```

### Flags
- `-vv` enables verbose output (recommended by subnet owner)
- `--test-mode` makes the validator only check your own uploaded rollouts

## Testing Precision Tuning

If you're running on non-A100 GPUs (H100, H200, RTX 4090, etc.), enable precision tuning:

```bash
# Miner with precision tuning
CUDA_VISIBLE_DEVICES=0 GRAIL_PRECISION_TUNING=1 grail mine

# Validator (verbose)
CUDA_VISIBLE_DEVICES=1 grail -vv validate --test-mode
```

### What Precision Tuning Does

**Level 1** (`GRAIL_PRECISION_TUNING=1`):
- Disables TF32 (19-bit precision that causes drift)
- Enables deterministic CUDA operations
- Uses highest precision for matrix multiplications

**Level 2** (`GRAIL_PRECISION_TUNING=2`) - More aggressive:
- All of Level 1, plus:
- `torch.use_deterministic_algorithms(True)`
- Forces eager attention (no flash/sdpa optimizations)
- Requires: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Optional: `NVIDIA_TF32_OVERRIDE=0` (system-level)

```bash
# Level 2 example
CUDA_VISIBLE_DEVICES=0 \
  GRAIL_PRECISION_TUNING=2 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  NVIDIA_TF32_OVERRIDE=0 \
  grail mine
```

This helps match A100 floating point behavior on other GPU architectures, but may not fully bridge the gap due to fundamental hardware differences.

## Interpreting Results

### Success
```
[proof_valid] SUCCESS | seq_len=XXX | verified_positions=16
```

### Failure - Sketch Mismatch
```
[proof_valid] Commitment verification FAILED at position X | sketch_diff=89 | tolerance=77
```

If `sketch_diff > tolerance`, the proof fails. Common causes:
1. **Checkpoint mismatch** - Miner and validator have different weights
2. **GPU precision** - Different architectures produce different floats
3. **Code bug** - Hidden state extraction differs

## Diagnostic Scripts

### 1. Verify Proof Locally
Tests miner vs validator hidden state extraction:
```bash
python scripts/verify_proof_locally.py --checkpoint /path/to/checkpoint
```

### 2. Debug Specific Rollout
Analyzes an actual rollout file:
```bash
python scripts/debug_proof_mismatch.py --latest --checkpoint /path/to/checkpoint
```

### 3. Check Checkpoint Integrity
Verifies checkpoint weights hash:
```bash
python scripts/check_checkpoint.py --checkpoint /path/to/checkpoint
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GRAIL_PRECISION_TUNING` | Cross-GPU float compatibility (1=basic, 2=aggressive) | `0` |
| `CUBLAS_WORKSPACE_CONFIG` | Required for Level 2 determinism | unset |
| `NVIDIA_TF32_OVERRIDE` | System-level TF32 disable | unset |
| `GRAIL_FORCE_EAGER_ATTENTION` | Force eager attention (auto with Level 2) | `0` |
| `CUDA_VISIBLE_DEVICES` | GPU assignment | all |

## Troubleshooting

### Proofs pass locally but fail on network
- GPU architecture difference between your machine and network validator
- Try `GRAIL_PRECISION_TUNING=1`
- Consider using A100 GPUs (explicitly recommended in docs)

### Proofs fail locally
- Check checkpoint window matches between miner and validator
- Run `check_checkpoint.py` to verify weights integrity
- Check logs for "CHECKPOINT WINDOW MISMATCH"

### Out of memory
- Run miner and validator sequentially instead of parallel
- Reduce `GRAIL_GENERATION_BATCH_SIZE`

## Notes

- Both miner and validator use the same `.env` configuration
- They share the checkpoint cache directory
- `--test-mode` only validates your own UID's files
