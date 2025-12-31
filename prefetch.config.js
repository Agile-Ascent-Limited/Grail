// prefetch.config.js
// PM2 ecosystem config for checkpoint prefetch mode
// Runs without GPU - continuously downloads new checkpoints to cache
//
// Usage:
//   pm2 start prefetch.config.js
//   pm2 logs prefetch
//
// This ensures checkpoints are pre-cached before miners need them,
// reducing startup time on slow nodes (e.g., A6000).

module.exports = {
  apps: [
    {
      name: "prefetch",
      script: "bash",
      args: "-c 'source /root/Grail/.venv/bin/activate && grail mine'",
      interpreter: "none",
      env: {
        GRAIL_PREFETCH_MODE: "1",
        GRAIL_PREFETCH_INTERVAL: "15",
        BT_WALLET_NAME: "c73",
        BT_HOTKEY_NAME: "c73-h10",
        SUBTENSOR_NETWORK: "finney",
        GRAIL_CACHE_ROOT: "/ephemeral/grail",
      }
    }
  ]
};
