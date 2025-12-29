// ecosystem-a100-multinode-hub.config.js - PM2 config for HUB node in multi-node setup
// This is the AGGREGATOR node that collects rollouts from all nodes and uploads
//
// MULTI-NODE ARCHITECTURE (4 nodes × 8 A100s = 32 GPUs):
//   Node 1 (HUB): 8x A100 miners → local staging → push to Redis → aggregate all → upload
//   Node 2: 8x A100 miners → local staging → push to Redis (hub uploads)
//   Node 3: 8x A100 miners → local staging → push to Redis (hub uploads)
//   Node 4: 8x A100 miners → local staging → push to Redis (hub uploads)
//
// REDIS SETUP:
//   Run Redis on the hub node (or accessible by all nodes):
//   docker run -d --name redis -p 6379:6379 redis:7-alpine
//
// ENVIRONMENT VARIABLES:
//   GRAIL_REDIS_URL=redis://hub-ip:6379/0  - Redis connection (required)
//   GRAIL_HUB_MODE=1                       - This node aggregates and uploads (hub only)
//   GRAIL_NODE_ID=node-1                   - Unique node identifier (auto-generated if not set)
//   GRAIL_TOTAL_NODES=4                    - Total number of nodes
//
// SETUP:
//   1. Start Redis: docker run -d --name redis -p 6379:6379 redis:7-alpine
//   2. pip install redis  (if not installed)
//   3. pm2 start ecosystem-a100-multinode-hub.config.js
//
// On other nodes, use ecosystem-a100-multinode-worker.config.js

module.exports = {
  apps: [
    // Worker 0 (LEADER) - GPU 0
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
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Multi-node settings (HUB)
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
        GRAIL_HUB_MODE: '1',
        GRAIL_NODE_ID: 'node-1',
        GRAIL_TOTAL_NODES: '2',
      },
      max_memory_restart: '80G',
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
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '1',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        // Multi-node: all workers need Redis for problem claiming
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
      },
      max_memory_restart: '80G',
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
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '2',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
      },
      max_memory_restart: '80G',
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
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '3',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
      },
      max_memory_restart: '80G',
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
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '4',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
      },
      max_memory_restart: '80G',
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
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '5',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
      },
      max_memory_restart: '80G',
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
        GRAIL_TOTAL_WORKERS: '8',
        CUDA_VISIBLE_DEVICES: '6',
        GRAIL_USE_VLLM: '1',
        GRAIL_USE_FLASH_ATTENTION: '0',
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-6-error.log',
      out_file: '/var/log/grail/worker-6-out.log',
      merge_logs: true,
    },
    // Worker 7 (FOLLOWER) - GPU 7
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
        GRAIL_GENERATION_BATCH_SIZE: '16',
        GRAIL_VLLM_GPU_MEMORY_UTIL: '0.55',
        GRAIL_VLLM_MAX_NUM_SEQS: '32',
        GRAIL_MINER_SAFETY_BLOCKS: '6',
        GRAIL_REDIS_URL: 'redis://localhost:6379/0',
      },
      max_memory_restart: '80G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/grail/worker-7-error.log',
      out_file: '/var/log/grail/worker-7-out.log',
      merge_logs: true,
    },
  ],
};
