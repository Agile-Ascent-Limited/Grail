// ecosystem.config.js - PM2 configuration for 8x A100 mining with vLLM
// Each worker spawns its own vLLM server on port 30000 + worker_id
//
// LEADER-FOLLOWER PATTERN:
//   Worker 0 (leader) starts immediately and initializes blockchain/checkpoints.
//   Workers 1-7 (followers) start after 30s delay to let leader init first.
//
// SETUP:
//   1. bash scripts/setup_vllm_env.sh     # Install vLLM environment
//   2. pm2 start ecosystem.config.js      # Start miners (vLLM auto-spawned)
//
// STOP:
//   pm2 stop all

// Delay in seconds for follower workers (1-7) to let leader initialize first
const FOLLOWER_DELAY_SECONDS = 30;

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
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',  // Extra buffer before deadline (5 blocks = ~60s)
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-0-error.log',
      out_file: '/var/log/grail/worker-0-out.log',
      merge_logs: true,
    },
    // Worker 1 (FOLLOWER) - GPU 1, starts after delay
    {
      name: 'grail-miner-1',
      script: 'bash',
      args: `-c "sleep ${FOLLOWER_DELAY_SECONDS} && exec .venv/bin/grail mine"`,
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '1',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '1',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-1-error.log',
      out_file: '/var/log/grail/worker-1-out.log',
      merge_logs: true,
    },
    // Worker 2 (FOLLOWER) - GPU 2, starts after delay
    {
      name: 'grail-miner-2',
      script: 'bash',
      args: `-c "sleep ${FOLLOWER_DELAY_SECONDS} && exec .venv/bin/grail mine"`,
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '2',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '2',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-2-error.log',
      out_file: '/var/log/grail/worker-2-out.log',
      merge_logs: true,
    },
    // Worker 3 (FOLLOWER) - GPU 3, starts after delay
    {
      name: 'grail-miner-3',
      script: 'bash',
      args: `-c "sleep ${FOLLOWER_DELAY_SECONDS} && exec .venv/bin/grail mine"`,
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '3',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '3',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-3-error.log',
      out_file: '/var/log/grail/worker-3-out.log',
      merge_logs: true,
    },
    // Worker 4 (FOLLOWER) - GPU 4, starts after delay
    {
      name: 'grail-miner-4',
      script: 'bash',
      args: `-c "sleep ${FOLLOWER_DELAY_SECONDS} && exec .venv/bin/grail mine"`,
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '4',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '4',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-4-error.log',
      out_file: '/var/log/grail/worker-4-out.log',
      merge_logs: true,
    },
    // Worker 5 (FOLLOWER) - GPU 5, starts after delay
    {
      name: 'grail-miner-5',
      script: 'bash',
      args: `-c "sleep ${FOLLOWER_DELAY_SECONDS} && exec .venv/bin/grail mine"`,
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '5',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '5',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-5-error.log',
      out_file: '/var/log/grail/worker-5-out.log',
      merge_logs: true,
    },
    // Worker 6 (FOLLOWER) - GPU 6, starts after delay
    {
      name: 'grail-miner-6',
      script: 'bash',
      args: `-c "sleep ${FOLLOWER_DELAY_SECONDS} && exec .venv/bin/grail mine"`,
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '6',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '6',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-6-error.log',
      out_file: '/var/log/grail/worker-6-out.log',
      merge_logs: true,
    },
    // Worker 7 (FOLLOWER) - GPU 7, starts after delay
    {
      name: 'grail-miner-7',
      script: 'bash',
      args: `-c "sleep ${FOLLOWER_DELAY_SECONDS} && exec .venv/bin/grail mine"`,
      interpreter: 'none',
      cwd: '/root/Grail',
      env: {
        GRAIL_WORKER_ID: '7',
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '7',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '8',
        GRAIL_MINER_SAFETY_BLOCKS: '5',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-7-error.log',
      out_file: '/var/log/grail/worker-7-out.log',
      merge_logs: true,
    },
  ],
};
