// ecosystem-testing.config.js - PM2 configuration for 7x GPU mining + 1x GPU validation (test mode)
// Each worker runs on its own GPU with vLLM (auto-spawned server)
//
// LEADER-FOLLOWER PATTERN:
//   Worker 0 (leader) initializes blockchain/checkpoints and signals ready via barrier file.
//   Workers 1-6 (followers) wait for leader's barrier signal before mining (no PM2 delay needed).
//
// LOCAL TESTING:
//   Validator runs on GPU 7 in --test-mode to validate your own rollouts locally.
//   This lets you verify proofs pass before network validation.
//
// H200 OPTIMIZATIONS (141GB HBM3e):
//   - GRAIL_VLLM_GPU_MEMORY_UTIL: 0.70 (70% for vLLM, leaves ~42GB for HF proof model)
//   - GRAIL_GENERATION_BATCH_SIZE: 16 (matches ROLLOUTS_PER_PROBLEM cap)
//   - GRAIL_VLLM_MAX_NUM_SEQS: 32 (16 prompts × 2x buffer)
//
// PRECISION TUNING (attempting A100 compatibility):
//   Level 1: GRAIL_PRECISION_TUNING=1
//     - Disables TF32, enables deterministic ops, highest matmul precision
//   Level 2: GRAIL_PRECISION_TUNING=2 (more aggressive)
//     - All of Level 1 plus torch.use_deterministic_algorithms(True)
//     - Forces eager attention (no flash/sdpa)
//     - Requires CUBLAS_WORKSPACE_CONFIG=:4096:8
//   Additional: NVIDIA_TF32_OVERRIDE=0 (system-level TF32 disable)
//
// SETUP:
//   pm2 start ecosystem-testing.config.js      # Start miners + validator
//
// STOP:
//   pm2 stop all

module.exports = {
  apps: [
    // Worker 0 (LEADER) - GPU 0, starts immediately
    {
      name: 'grail-miner-0',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '0',
        GRAIL_TOTAL_WORKERS: '7',
        CUDA_VISIBLE_DEVICES: '0',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.70',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Level 1 precision tuning (Level 2 breaks vLLM/HF compatibility)
        GRAIL_PRECISION_TUNING: '1',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-0-error.log',
      out_file: '/var/log/grail/worker-0-out.log',
      merge_logs: true,
    },
    // Worker 1 (FOLLOWER) - GPU 1
    {
      name: 'grail-miner-1',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '1',
        GRAIL_TOTAL_WORKERS: '7',
        CUDA_VISIBLE_DEVICES: '1',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.70',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Level 1 precision tuning (Level 2 breaks vLLM/HF compatibility)
        GRAIL_PRECISION_TUNING: '1',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-1-error.log',
      out_file: '/var/log/grail/worker-1-out.log',
      merge_logs: true,
    },
    // Worker 2 (FOLLOWER) - GPU 2
    {
      name: 'grail-miner-2',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '2',
        GRAIL_TOTAL_WORKERS: '7',
        CUDA_VISIBLE_DEVICES: '2',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.70',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Level 1 precision tuning (Level 2 breaks vLLM/HF compatibility)
        GRAIL_PRECISION_TUNING: '1',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-2-error.log',
      out_file: '/var/log/grail/worker-2-out.log',
      merge_logs: true,
    },
    // Worker 3 (FOLLOWER) - GPU 3
    {
      name: 'grail-miner-3',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '3',
        GRAIL_TOTAL_WORKERS: '7',
        CUDA_VISIBLE_DEVICES: '3',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.70',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Level 1 precision tuning (Level 2 breaks vLLM/HF compatibility)
        GRAIL_PRECISION_TUNING: '1',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-3-error.log',
      out_file: '/var/log/grail/worker-3-out.log',
      merge_logs: true,
    },
    // Worker 4 (FOLLOWER) - GPU 4
    {
      name: 'grail-miner-4',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '4',
        GRAIL_TOTAL_WORKERS: '7',
        CUDA_VISIBLE_DEVICES: '4',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.70',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Level 1 precision tuning (Level 2 breaks vLLM/HF compatibility)
        GRAIL_PRECISION_TUNING: '1',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-4-error.log',
      out_file: '/var/log/grail/worker-4-out.log',
      merge_logs: true,
    },
    // Worker 5 (FOLLOWER) - GPU 5
    {
      name: 'grail-miner-5',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '5',
        GRAIL_TOTAL_WORKERS: '7',
        CUDA_VISIBLE_DEVICES: '5',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.70',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Level 1 precision tuning (Level 2 breaks vLLM/HF compatibility)
        GRAIL_PRECISION_TUNING: '1',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-5-error.log',
      out_file: '/var/log/grail/worker-5-out.log',
      merge_logs: true,
    },
    // Worker 6 (FOLLOWER) - GPU 6
    {
      name: 'grail-miner-6',
      script: '.venv/bin/grail',
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '6',
        GRAIL_TOTAL_WORKERS: '7',
        CUDA_VISIBLE_DEVICES: '6',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.70',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Level 1 precision tuning (Level 2 breaks vLLM/HF compatibility)
        GRAIL_PRECISION_TUNING: '1',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-6-error.log',
      out_file: '/var/log/grail/worker-6-out.log',
      merge_logs: true,
    },
    // VALIDATOR (TEST MODE) - GPU 7
    // Validates your own rollouts locally before network validation
    // Uses -vv for verbose output as recommended by subnet owner
    {
      name: 'grail-validator',
      script: '.venv/bin/grail',
      args: '-vv validate --test-mode',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        CUDA_VISIBLE_DEVICES: '7',
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/validator-error.log',
      out_file: '/var/log/grail/validator-out.log',
      merge_logs: true,
    },
  ],
};
