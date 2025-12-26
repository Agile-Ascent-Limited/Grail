// ecosystem.config.js - PM2 configuration for 8x H200 mining with vLLM backend
// Each worker runs on its own GPU with vLLM (auto-spawned server)
//
// LEADER-FOLLOWER PATTERN:
//   Worker 0 (leader) initializes blockchain/checkpoints and signals ready via barrier file.
//   Workers 1-7 (followers) wait for leader's barrier signal before mining (no PM2 delay needed).
//
// H200 OPTIMIZATIONS (141GB HBM3e):
//   - GRAIL_VLLM_GPU_MEMORY_UTIL: 0.85 (use 85% for KV cache, ~120GB)
//   - GRAIL_GENERATION_BATCH_SIZE: 64 (larger batches for H200 throughput)
//   - GRAIL_VLLM_MAX_NUM_SEQS: 128 (allow more concurrent sequences)
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
      args: 'mine',
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '0',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '0',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-0-error.log',
      out_file: '/var/log/grail/worker-0-out.log',
      merge_logs: true,
    },
    // Worker 1 (FOLLOWER) - GPU 1, waits for leader barrier
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
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-1-error.log',
      out_file: '/var/log/grail/worker-1-out.log',
      merge_logs: true,
    },
    // Worker 2 (FOLLOWER) - GPU 2, waits for leader barrier
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
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-2-error.log',
      out_file: '/var/log/grail/worker-2-out.log',
      merge_logs: true,
    },
    // Worker 3 (FOLLOWER) - GPU 3, waits for leader barrier
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
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-3-error.log',
      out_file: '/var/log/grail/worker-3-out.log',
      merge_logs: true,
    },
    // Worker 4 (FOLLOWER) - GPU 4, waits for leader barrier
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
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-4-error.log',
      out_file: '/var/log/grail/worker-4-out.log',
      merge_logs: true,
    },
    // Worker 5 (FOLLOWER) - GPU 5, waits for leader barrier
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
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-5-error.log',
      out_file: '/var/log/grail/worker-5-out.log',
      merge_logs: true,
    },
    // Worker 6 (FOLLOWER) - GPU 6, waits for leader barrier
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
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-6-error.log',
      out_file: '/var/log/grail/worker-6-out.log',
      merge_logs: true,
    },
    // Worker 7 (FOLLOWER) - GPU 7, waits for leader barrier
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
        GRAIL_USE_FLASH_ATTENTION: '0',  // vLLM has flash-attn built-in
        GRAIL_GENERATION_BATCH_SIZE: '64',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.85',
        GRAIL_VLLM_MAX_NUM_SEQS: '128',
        GRAIL_MINER_SAFETY_BLOCKS: '4',  // Safe buffer: 4 blocks = ~48s
      },
      max_memory_restart: '140G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-7-error.log',
      out_file: '/var/log/grail/worker-7-out.log',
      merge_logs: true,
    },
  ],
};
