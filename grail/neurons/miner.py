from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import traceback
from types import SimpleNamespace

import bittensor as bt
import torch

# ============================================================================
# PRECISION SETTINGS FOR CROSS-GPU COMPATIBILITY
# These settings help ensure floating point consistency across GPU architectures
# (A100, H100, H200, RTX 4090, etc.) by disabling GPU-specific optimizations
# that can cause minor numerical differences.
#
# Level 1 (GRAIL_PRECISION_TUNING=1): Basic precision tuning
#   - Disables TF32, enables deterministic ops, highest matmul precision
#
# Level 2 (GRAIL_PRECISION_TUNING=2): Aggressive precision tuning
#   - All of Level 1 plus:
#   - torch.use_deterministic_algorithms(True)
#   - Forces eager attention (no flash/sdpa optimizations)
#   - Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable
# ============================================================================
_PRECISION_LEVEL = os.getenv("GRAIL_PRECISION_TUNING", "0")
_ENABLE_PRECISION_TUNING = _PRECISION_LEVEL.lower() in ("1", "2", "true", "yes")
_AGGRESSIVE_PRECISION = _PRECISION_LEVEL == "2"

# Download-only mode: just download latest checkpoint and exit
# Useful for pre-staging checkpoints on new nodes before joining the cluster
_DOWNLOAD_ONLY = os.getenv("GRAIL_DOWNLOAD_ONLY", "0").lower() in ("1", "true", "yes")

# Prefetch mode: continuously watch for new checkpoints and pre-download them
# Runs without GPU - perfect for slow nodes that need checkpoints pre-staged
# Use: GRAIL_PREFETCH_MODE=1 grail mine
_PREFETCH_MODE = os.getenv("GRAIL_PREFETCH_MODE", "0").lower() in ("1", "true", "yes")
_PREFETCH_INTERVAL = int(os.getenv("GRAIL_PREFETCH_INTERVAL", "30"))  # Check every N seconds
# Prefetch FULL checkpoints only (for cold start optimization across workers)
# Default "0" = prefetch ALL checkpoints (FULL + DELTA) to keep cache warm
_PREFETCH_FULL_ONLY = os.getenv("GRAIL_PREFETCH_FULL_ONLY", "0").lower() in ("1", "true", "yes")

if _ENABLE_PRECISION_TUNING:
    # Disable TF32 - uses 19-bit precision instead of 23-bit, causes drift
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Force deterministic operations where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use highest precision for matrix multiplications
    torch.set_float32_matmul_precision('highest')

    if _AGGRESSIVE_PRECISION:
        # Level 2: Most aggressive determinism
        # NOTE: Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 to be set
        try:
            torch.use_deterministic_algorithms(True)
            # Force eager attention by setting environment variable
            os.environ["GRAIL_USE_FLASH_ATTENTION"] = "0"
            os.environ["GRAIL_FORCE_EAGER_ATTENTION"] = "1"
            logging.getLogger(__name__).info(
                "PRECISION TUNING LEVEL 2: deterministic_algorithms=True, eager_attention=True"
            )
        except RuntimeError as e:
            logging.getLogger(__name__).warning(
                f"Could not enable use_deterministic_algorithms: {e}. "
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable."
            )
    else:
        logging.getLogger(__name__).info(
            "PRECISION TUNING LEVEL 1: TF32=disabled, deterministic=True, matmul_precision=highest"
        )

from grail.cli.mine import (
    MiningTimers,
    generate_rollouts_for_window,
    get_conf,
    get_window_randomness,
    has_time_for_next_generation,
    upload_inferences_with_metrics,
)
from grail.environments.execution import (
    CodeExecutionPool,
    set_global_execution_pool,
)
from grail.infrastructure.chain import GrailChainManager, fetch_trainer_bucket_sync
from grail.trainer.config import EvalConfig
from grail.trainer.inference_server import ServerConfig, VLLMServerManager
from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.credentials import load_r2_credentials
from grail.infrastructure.worker_barrier import (
    WorkerBarrier, RolloutStaging, get_redis_rollout_aggregator,
)
from grail.infrastructure.worker_config import WorkerConfig, log_multi_gpu_setup
from grail.shared.schemas import Bucket
from grail.model.provider import clear_model_and_tokenizer, get_model, get_tokenizer
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig
from grail.shared.constants import TRAINER_UID, WINDOW_LENGTH
from grail.shared.subnet import get_own_uid_on_subnet
from grail.shared.window_utils import (
    WindowWaitTracker,
    calculate_next_window,
    log_window_wait_initial,
    log_window_wait_periodic,
)
from datetime import datetime

from .base import BaseNeuron

logger = logging.getLogger(__name__)


def _create_vllm_server_manager(
    checkpoint_path: str,
    worker_id: int,
    gpu_memory_util: float = 0.50,
) -> VLLMServerManager | None:
    """Create a vLLM server manager for the miner if GRAIL_USE_VLLM=1.

    The server port is calculated as: base_port + worker_id
    Default base port is 30000, so worker 0 uses 30000, worker 1 uses 30001, etc.

    Args:
        checkpoint_path: Path to the model checkpoint
        worker_id: Worker ID for port calculation
        gpu_memory_util: GPU memory fraction for vLLM (default 0.50 to leave room for proof model)

    Returns:
        VLLMServerManager if vLLM is enabled, None otherwise
    """
    use_vllm = os.getenv("GRAIL_USE_VLLM", "0") == "1"
    if not use_vllm:
        return None

    base_port = int(os.getenv("GRAIL_VLLM_BASE_PORT", "30000"))
    port = base_port + worker_id

    # Create server config
    server_config = ServerConfig(
        host="127.0.0.1",
        port=port,
        timeout_s=300.0,
        trust_remote_code=True,
        dtype="bfloat16",
        model_path=checkpoint_path,
    )

    # Create eval config for vLLM settings
    eval_config = EvalConfig(
        vllm_gpu_memory_utilization=gpu_memory_util,
        vllm_max_model_len=int(os.getenv("GRAIL_VLLM_MAX_MODEL_LEN", "4096")),
        vllm_max_num_seqs=int(os.getenv("GRAIL_VLLM_MAX_NUM_SEQS", "32")),
        stream_server_logs=False,  # Don't stream logs to keep output clean
    )

    # Get vLLM Python path
    vllm_python = os.getenv(
        "GRAIL_VLLM_PYTHON",
        "tools/vllm-server/.venv/bin/python",
    )

    manager = VLLMServerManager(
        config=server_config,
        eval_config=eval_config,
        python_executable=vllm_python,
    )

    logger.info(
        "Created vLLM server manager for worker %d (port %d, gpu_mem=%.0f%%)",
        worker_id,
        port,
        gpu_memory_util * 100,
    )

    return manager


class MinerNeuron(BaseNeuron):
    """Runs the mining loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True) -> None:
        super().__init__()
        self.use_drand = use_drand

    # (heartbeat is now handled by BaseNeuron.heartbeat())

    async def run(self) -> None:
        """Main mining loop mirrored from the CLI implementation."""
        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        logger.info(f"🔑 Miner hotkey: {wallet.hotkey.ss58_address}")

        # Log multi-GPU and worker configuration
        log_multi_gpu_setup()
        worker_config = WorkerConfig.from_env()

        # Model and tokenizer will be loaded from checkpoint
        model = None
        tokenizer = None
        current_checkpoint_window: int | None = None
        window_wait_tracker = WindowWaitTracker(log_interval_secs=120)
        vllm_server: VLLMServerManager | None = None

        async def _run() -> None:
            nonlocal model, tokenizer, current_checkpoint_window, vllm_server
            last_window_start = -1
            timers = MiningTimers()

            # Load R2 credentials (all workers need this for upload)
            try:
                credentials = load_r2_credentials()
                logger.info("✅ Loaded R2 credentials")
            except Exception as e:
                logger.error(f"Failed to load R2 credentials: {e}")
                raise

            # Initialize heartbeat (watchdog will monitor for stalls)
            self.heartbeat()
            logger.info("✅ Initialized watchdog heartbeat")

            # Initialize fast code execution pool for MBPP/HumanEval environments
            # This eliminates ~6s spawn overhead per code execution (50-100x speedup)
            execution_pool: CodeExecutionPool | None = None
            try:
                execution_pool = CodeExecutionPool(
                    num_workers=4,  # Fewer workers for miner (less parallel load)
                    max_tasks_per_child=50,
                )
                execution_pool.start()
                set_global_execution_pool(execution_pool)
                logger.info("✅ Fast code execution pool initialized: %d workers", 4)
            except Exception as e:
                logger.warning("⚠️ Failed to init execution pool, using slow path: %s", e)
                execution_pool = None

            def _cleanup_execution_pool() -> None:
                nonlocal execution_pool
                if execution_pool is not None:
                    try:
                        logger.info("Shutting down code execution pool...")
                        set_global_execution_pool(None)
                        execution_pool.shutdown()
                        execution_pool = None
                        logger.info("✅ Code execution pool shutdown complete")
                    except Exception as e:
                        logger.warning(f"Error shutting down execution pool: {e}")

            self.register_shutdown_callback(_cleanup_execution_pool)
            self.heartbeat()

            # Create worker barrier for leader-follower synchronization
            barrier = WorkerBarrier(
                cache_root=default_checkpoint_cache_root(),
                worker_id=worker_config.worker_id,
                total_workers=worker_config.total_workers,
            )

            # Create rollout staging for multi-worker aggregation
            # In multi-worker mode, all workers save locally, leader aggregates and uploads
            rollout_staging: RolloutStaging | None = None
            if worker_config.total_workers > 1:
                rollout_staging = RolloutStaging(
                    cache_root=default_checkpoint_cache_root(),
                    worker_id=worker_config.worker_id,
                    total_workers=worker_config.total_workers,
                )
                logger.info(
                    "📦 Multi-worker mode: using rollout staging (worker %d/%d)",
                    worker_config.worker_id, worker_config.total_workers,
                )

            # Redis-based rollout aggregation (per-worker direct push to Redis)
            # Each worker pushes their own rollouts; hub (worker 0) aggregates and uploads
            redis_aggregator = get_redis_rollout_aggregator(
                worker_id=worker_config.worker_id,
                total_workers=worker_config.total_workers,
            )
            if redis_aggregator is not None:
                if redis_aggregator.is_hub:
                    logger.info(
                        "🌐 Redis HUB mode: worker %d will aggregate from %d workers × %d nodes and upload",
                        worker_config.worker_id,
                        redis_aggregator.total_workers,
                        redis_aggregator.total_nodes,
                    )
                else:
                    logger.info(
                        "🌐 Redis mode: worker %s will push rollouts directly to Redis",
                        redis_aggregator.worker_key,
                    )

            netuid = int(get_conf("BT_NETUID", get_conf("NETUID", 200)))
            chain_manager = None  # Only leader initializes this

            if barrier.is_leader:
                # LEADER (worker 0): Do blockchain initialization
                logger.info("🎯 Worker 0 is LEADER - initializing blockchain connections...")

                # Determine if this is the hub (node-1 worker-0) or a non-hub leader (node-2+ worker-0)
                is_hub = redis_aggregator is not None and redis_aggregator.is_hub

                # Download-only mode: fetch trainer bucket directly without full chain manager
                if _DOWNLOAD_ONLY:
                    logger.info("📥 Download-only mode: fetching trainer bucket from chain...")
                    trainer_bucket = fetch_trainer_bucket_sync(netuid, TRAINER_UID)
                    if trainer_bucket:
                        logger.info(f"📥 Got trainer bucket: {trainer_bucket.name.strip()}")
                        checkpoint_credentials = trainer_bucket
                    else:
                        logger.warning("📥 Could not fetch trainer bucket, using local credentials")
                        checkpoint_credentials = credentials
                    subtensor = await self.get_subtensor()
                    self.heartbeat()
                else:
                    # NON-HUB LEADERS: Wait for hub to be ready BEFORE heavy blockchain init
                    # This ensures node-2+ workers don't start slow init while hub is still starting
                    # Hub caches block early, so once hub is ready, block cache is available
                    if redis_aggregator is not None and not is_hub:
                        logger.info("⏳ Non-hub leader waiting for hub to be ready...")
                        wait_start = time.monotonic()
                        max_wait = 120.0  # 2 minutes max wait for hub
                        while not redis_aggregator.is_hub_ready():
                            elapsed = time.monotonic() - wait_start
                            if elapsed > max_wait:
                                logger.warning(
                                    "⚠️ Hub not ready after %.0fs - proceeding anyway",
                                    elapsed,
                                )
                                break
                            if int(elapsed) % 10 == 0 and elapsed > 0:
                                logger.info(
                                    "⏳ Still waiting for hub (%.0fs)...",
                                    elapsed,
                                )
                            await asyncio.sleep(1.0)
                        else:
                            elapsed = time.monotonic() - wait_start
                            logger.info("✅ Hub ready after %.1fs - proceeding with init", elapsed)

                    # Get subtensor and metagraph for chain manager
                    subtensor = await self.get_subtensor()
                    self.heartbeat()

                    # HUB EARLY CACHE: If this is the hub (node-1 worker-0), cache block to Redis
                    # immediately so other nodes can start their mining loop without waiting.
                    # This reduces cross-node startup lag significantly.
                    if is_hub:
                        try:
                            early_block = await subtensor.get_current_block()
                            redis_aggregator.cache_current_block(early_block)
                            logger.info("🌐 Hub early cache: block %d → Redis (other nodes can start)", early_block)
                        except Exception as e:
                            logger.warning("Failed to early-cache block to Redis: %s", e)

                    metagraph = await subtensor.metagraph(netuid)
                    self.heartbeat()

                    # Initialize chain manager for credential commitments
                    config = SimpleNamespace(netuid=netuid)
                    chain_manager = GrailChainManager(config, wallet, metagraph, subtensor, credentials)
                    await chain_manager.initialize()
                    logger.info("✅ Initialized chain manager and committed read credentials")
                    # Ensure background chain worker stops on shutdown
                    self.register_shutdown_callback(chain_manager.stop)
                    self.heartbeat()

                    # Use trainer UID's committed read credentials for checkpoints
                    trainer_bucket = chain_manager.get_bucket(TRAINER_UID)
                    if trainer_bucket is not None:
                        logger.info(f"✅ Using trainer UID {TRAINER_UID} bucket for checkpoints")
                        checkpoint_credentials = trainer_bucket
                    else:
                        logger.warning(
                            f"⚠️ Trainer UID {TRAINER_UID} bucket not found, using local credentials"
                        )
                        checkpoint_credentials = credentials
                        trainer_bucket = None

                    # Signal ready to followers with shared data
                    barrier.signal_ready({
                        "trainer_bucket": trainer_bucket.model_dump() if trainer_bucket else None,
                        "netuid": netuid,
                    })
                    # Register cleanup on shutdown
                    self.register_shutdown_callback(barrier.cleanup)
                    logger.info("✅ Leader signaled ready to followers")

                    # Hub signals ready to Redis for cross-node coordination
                    if is_hub:
                        redis_aggregator.signal_hub_ready()

            else:
                # FOLLOWER (workers 1-7): Wait for leader, skip blockchain init
                # In download-only mode, fetch trainer bucket and skip leader wait
                if _DOWNLOAD_ONLY:
                    logger.info("📥 Download-only mode: fetching trainer bucket from chain...")
                    trainer_bucket = fetch_trainer_bucket_sync(netuid, TRAINER_UID)
                    if trainer_bucket:
                        logger.info(f"📥 Got trainer bucket: {trainer_bucket.name.strip()}")
                        checkpoint_credentials = trainer_bucket
                    else:
                        logger.warning("📥 Could not fetch trainer bucket, using local credentials")
                        checkpoint_credentials = credentials
                    subtensor = await self.get_subtensor()
                    self.heartbeat()
                else:
                    logger.info(
                        "⏳ Worker %d is FOLLOWER - waiting for leader...",
                        worker_config.worker_id,
                    )

                    try:
                        shared_data = await barrier.wait_for_leader(timeout=300)
                        logger.info("✅ Received leader signal, proceeding with mining")
                    except TimeoutError:
                        logger.error(
                            "❌ Timed out waiting for leader (worker 0). "
                            "Ensure worker 0 is running and healthy."
                        )
                        raise

                    # Reconstruct trainer_bucket from shared data
                    trainer_bucket_data = shared_data.get("trainer_bucket")
                    if trainer_bucket_data:
                        trainer_bucket = Bucket.model_validate(trainer_bucket_data)
                        checkpoint_credentials = trainer_bucket
                        logger.info(f"✅ Using trainer bucket from leader for checkpoints")
                    else:
                        logger.warning("⚠️ No trainer bucket from leader, using local credentials")
                        checkpoint_credentials = credentials

                    # Followers still need subtensor for block checks in mining loop
                    subtensor = await self.get_subtensor()
                    self.heartbeat()

            checkpoint_manager = CheckpointManager(
                cache_root=default_checkpoint_cache_root(),
                credentials=checkpoint_credentials,
                keep_limit=2,  # Keep only current + previous window
            )

            # Download-only mode: download latest checkpoint and exit
            # Use: GRAIL_DOWNLOAD_ONLY=1 grail mine
            if _DOWNLOAD_ONLY:
                logger.info("📥 Download-only mode: fetching latest checkpoint...")
                try:
                    # Get current block to find latest checkpoint
                    subtensor = await self.get_subtensor()
                    current_block = await subtensor.get_current_block()
                    logger.info(f"📥 Current block: {current_block}")

                    # Discover latest ready checkpoint
                    checkpoint_window = await checkpoint_manager.get_latest_ready_checkpoint(
                        current_block
                    )

                    if checkpoint_window is None:
                        logger.error("❌ No checkpoint available to download")
                        return

                    logger.info(f"📥 Found checkpoint: {checkpoint_window}")

                    # Download it
                    checkpoint_path = await checkpoint_manager.get_checkpoint(checkpoint_window)

                    if checkpoint_path:
                        logger.info(f"✅ Checkpoint downloaded to: {checkpoint_path}")
                        logger.info("📥 Download-only mode complete - exiting")
                    else:
                        logger.error("❌ Checkpoint download failed")
                except Exception as e:
                    logger.error(f"❌ Download-only mode failed: {e}")
                    traceback.print_exc()
                return  # Exit after download

            # Prefetch mode: continuously watch for and download new checkpoints
            # Runs without GPU - perfect for slow nodes that need pre-staging
            # Use: GRAIL_PREFETCH_MODE=1 grail mine
            if _PREFETCH_MODE:
                logger.info("🔄 Prefetch mode: continuously watching for new checkpoints...")
                logger.info(f"🔄 Check interval: {_PREFETCH_INTERVAL}s (set GRAIL_PREFETCH_INTERVAL to change)")
                if _PREFETCH_FULL_ONLY:
                    logger.info("🔄 FULL-only mode: will only prefetch FULL checkpoints (cold start optimized)")

                last_downloaded_window: int | None = None

                while not self.stop_event.is_set():
                    try:
                        # Get current block to find latest checkpoint
                        subtensor = await self.get_subtensor()
                        current_block = await subtensor.get_current_block()
                        self.heartbeat()

                        # Discover latest checkpoint (FULL-only or any)
                        if _PREFETCH_FULL_ONLY:
                            checkpoint_window = await checkpoint_manager.get_latest_full_checkpoint(
                                current_block
                            )
                        else:
                            checkpoint_window = await checkpoint_manager.get_latest_ready_checkpoint(
                                current_block
                            )

                        if checkpoint_window is None:
                            logger.debug("No checkpoint available yet, waiting...")
                        elif checkpoint_window == last_downloaded_window:
                            logger.debug(
                                f"Checkpoint {checkpoint_window} already downloaded, waiting for new one..."
                            )
                        else:
                            # New checkpoint available - download it
                            logger.info(
                                f"📥 New checkpoint detected: {checkpoint_window} (block {current_block})"
                            )

                            checkpoint_path = await checkpoint_manager.get_checkpoint(checkpoint_window)

                            if checkpoint_path:
                                logger.info(f"✅ Prefetched checkpoint: {checkpoint_path}")
                                last_downloaded_window = checkpoint_window
                            else:
                                logger.warning(f"⚠️ Failed to download checkpoint {checkpoint_window}")

                        # Wait before checking again
                        await asyncio.sleep(_PREFETCH_INTERVAL)

                    except asyncio.CancelledError:
                        logger.info("🔄 Prefetch mode cancelled")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Prefetch error (will retry): {e}")
                        await asyncio.sleep(_PREFETCH_INTERVAL)

                logger.info("🔄 Prefetch mode stopped")
                return  # Exit - prefetch mode doesn't mine

            # Initialize monitoring for mining operations (all workers)
            monitor = get_monitoring_manager()
            if monitor:
                mining_config = MonitoringConfig.for_mining(wallet.name)
                uid = None
                if barrier.is_leader:
                    # Leader can look up UID from metagraph
                    try:
                        subtensor_for_uid = await self.get_subtensor()
                        self.heartbeat()
                        uid = await get_own_uid_on_subnet(
                            subtensor_for_uid, netuid, wallet.hotkey.ss58_address
                        )
                        self.heartbeat()
                    except Exception:
                        pass
                run_name = f"miner-{uid}" if uid is not None else f"mining_{wallet.name}"
                run_id = await monitor.start_run(run_name, mining_config.get("hyperparameters", {}))
                self.heartbeat()
                logger.info(f"Started monitoring run: {run_id} (name={run_name})")

            # Persistent prefetch task reference - survives across loop iterations
            # This ensures we can await it and check its result at checkpoint loading time
            prefetch_task: asyncio.Task | None = None

            while not self.stop_event.is_set():
                try:
                    # Track if checkpoint changed this iteration (for cleanup timing)
                    checkpoint_changed_this_window = False

                    # Update heartbeat at start of each iteration
                    self.heartbeat()

                    # Use shared subtensor from base class
                    subtensor = await self.get_subtensor()

                    # Get current block: ONLY hub queries blockchain, all others read from Redis
                    # This ensures perfect sync across all workers on all nodes
                    is_hub = redis_aggregator is not None and redis_aggregator.is_hub

                    if is_hub:
                        # Hub (node-1 worker-0) queries blockchain and caches to Redis
                        current_block = await subtensor.get_current_block()
                        redis_aggregator.cache_current_block(current_block)
                        # Log periodically to avoid spam (every 10 blocks)
                        if current_block % 10 == 0:
                            logger.info("🌐 Hub: block %d → Redis", current_block)
                    elif redis_aggregator is not None:
                        # All other workers: read from Redis only (never RPC)
                        # Wait for hub to populate cache - ensures perfect sync
                        current_block = redis_aggregator.get_cached_block()
                        wait_count = 0
                        while current_block is None:
                            wait_count += 1
                            if wait_count == 1:
                                logger.info("⏳ Waiting for hub to cache block to Redis...")
                            elif wait_count % 10 == 0:
                                logger.warning(
                                    "⚠️ Still waiting for Redis block cache (%ds) - is hub running?",
                                    wait_count,
                                )
                            await asyncio.sleep(1)
                            current_block = redis_aggregator.get_cached_block()
                        if wait_count > 0:
                            logger.info("✅ Got block %d from Redis after %ds", current_block, wait_count)
                        else:
                            # Log periodically to show block is updating (every 10 blocks)
                            if current_block % 10 == 0:
                                logger.info("📡 Redis block: %d", current_block)
                    else:
                        # No Redis configured - use RPC directly (single-node mode)
                        current_block = await subtensor.get_current_block()

                    window_start = self.calculate_window(current_block)

                    # Set monitoring context for metrics (use block_number for x-axis)
                    if monitor:
                        monitor.set_block_context(current_block, None)

                    if window_start <= last_window_start:
                        if window_wait_tracker.should_log_initial():
                            log_window_wait_initial(
                                current_block=current_block,
                                last_processed_window=last_window_start,
                                window_length=WINDOW_LENGTH,
                            )
                        elif window_wait_tracker.should_log_periodic():
                            next_window = calculate_next_window(last_window_start, WINDOW_LENGTH)
                            log_window_wait_periodic(
                                next_window=next_window,
                                elapsed_seconds=window_wait_tracker.get_elapsed_seconds(),
                                current_block=current_block,
                            )

                        await asyncio.sleep(2)
                        continue

                    # Window is available - reset tracker
                    window_wait_tracker.reset()

                    # Hub signals window to Redis for cross-node sync
                    # Local leader signals to file-based barrier for same-node sync
                    if is_hub:
                        redis_aggregator.signal_window_start(window_start)
                        # Clear any previous stop signal from last window
                        redis_aggregator.clear_generation_stop()
                        # Clear stale checkpoint so non-hub waits for fresh discovery
                        redis_aggregator.clear_checkpoint_broadcast()
                    if barrier.is_leader:
                        barrier.signal_current_window(window_start)

                    # NON-HUB LEADERS: Do independent R2 discovery in parallel with hub.
                    # Previously we waited for hub's checkpoint broadcast, but that adds latency.
                    # Both nodes will find the same checkpoint (R2 discovery is deterministic),
                    # so there's no benefit to waiting. Parallel discovery is faster.
                    hub_checkpoint_hint: int | None = None
                    if barrier.is_leader and redis_aggregator is not None and not is_hub:
                        # Quick check: if hub already broadcast, use it (no wait, just check once)
                        redis_ckpt, _ = redis_aggregator.get_redis_checkpoint()
                        if redis_ckpt is not None:
                            if redis_ckpt != current_checkpoint_window:
                                logger.info(
                                    f"📡 Non-hub leader: Hub already broadcast checkpoint {redis_ckpt}, "
                                    f"using for parallel download"
                                )
                            else:
                                logger.info(
                                    f"📡 Non-hub leader: Hub already broadcast unchanged checkpoint {redis_ckpt}"
                                )
                            hub_checkpoint_hint = redis_ckpt
                        else:
                            # Hub hasn't broadcast yet - do independent discovery (parallel with hub)
                            logger.info(
                                "📡 Non-hub leader: No hub checkpoint yet, doing independent R2 discovery"
                            )

                    # Followers: check if leader is downloading a checkpoint BEFORE
                    # we discover/load checkpoints. This ensures all workers sync.
                    if not barrier.is_leader:
                        # Fast path: check Redis for cross-node checkpoint info (instant)
                        if redis_aggregator is not None:
                            redis_ckpt, _ = redis_aggregator.get_redis_checkpoint()
                            if redis_ckpt is not None and redis_ckpt != current_checkpoint_window:
                                logger.info(
                                    f"📡 Redis has checkpoint {redis_ckpt} (current={current_checkpoint_window})"
                                )
                        # Fall back to file-based barrier check for local workers
                        leader_ckpt = await barrier.wait_for_checkpoint_sync(
                            current_checkpoint_window, timeout=180.0
                        )
                        # If leader has a newer checkpoint ready or downloading, we'll get it
                        # through checkpoint_manager below. Just log for visibility.
                        if leader_ckpt is not None and leader_ckpt != current_checkpoint_window:
                            logger.info(
                                f"⏳ Leader has checkpoint {leader_ckpt}, will sync..."
                            )

                    # PREFETCH COMPLETION: If prefetch task exists, await it with timeout
                    # This ensures prefetch benefits are realized before we try to download
                    if prefetch_task is not None and not prefetch_task.done():
                        logger.info("⏳ Waiting for prefetch to complete (max 10s)...")
                        try:
                            await asyncio.wait_for(prefetch_task, timeout=10.0)
                            logger.info("✅ Prefetch completed - checkpoint should be cached")
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ Prefetch timed out - will download checkpoint normally")
                            prefetch_task.cancel()
                            try:
                                await prefetch_task
                            except asyncio.CancelledError:
                                pass
                        except Exception as e:
                            logger.debug(f"Prefetch failed: {e}")
                        prefetch_task = None
                    elif prefetch_task is not None and prefetch_task.done():
                        # Prefetch already completed - check for errors
                        try:
                            prefetch_task.result()
                            logger.debug("Prefetch already completed successfully")
                        except Exception as e:
                            logger.debug(f"Prefetch completed with error: {e}")
                        prefetch_task = None

                    # Discover latest ready checkpoint (before current window)
                    # Non-hub leaders can skip R2 discovery if hub already told us which checkpoint
                    if hub_checkpoint_hint is not None:
                        checkpoint_window = hub_checkpoint_hint
                        logger.info(
                            f"📡 Non-hub leader: Using hub's checkpoint {hub_checkpoint_hint} "
                            f"(skipped R2 discovery)"
                        )
                    else:
                        # Standard discovery from R2 (hub and single-node mode)
                        checkpoint_window = await checkpoint_manager.get_latest_ready_checkpoint(
                            window_start
                        )

                    # Load checkpoint if discovered and different from current
                    if checkpoint_window is not None and checkpoint_window >= 0:
                        if current_checkpoint_window != checkpoint_window:
                            # Leader signals to followers that it's downloading
                            if barrier.is_leader:
                                barrier.signal_checkpoint_downloading(checkpoint_window)

                            # Hub broadcasts checkpoint to Redis IMMEDIATELY so other nodes
                            # can skip discovery and start parallel downloads
                            if is_hub and redis_aggregator is not None:
                                redis_aggregator.broadcast_checkpoint(
                                    checkpoint_window, f"downloading-{checkpoint_window}"
                                )
                                logger.info(
                                    f"🌐 Hub: Broadcast checkpoint {checkpoint_window} to Redis "
                                    f"(other nodes can start parallel download)"
                                )

                            # Time checkpoint download/retrieval
                            timer_ctx = (
                                monitor.timer("profiling/checkpoint_download")
                                if monitor
                                else contextlib.nullcontext()
                            )
                            with timer_ctx:
                                checkpoint_path = await checkpoint_manager.get_checkpoint(
                                    checkpoint_window
                                )
                            self.heartbeat()

                            if checkpoint_path is not None:
                                logger.info(
                                    "[miner] Loading checkpoint for window %s from %s",
                                    checkpoint_window,
                                    checkpoint_path,
                                )
                                try:
                                    # Pre-load cleanup to prevent VRAM growth when swapping
                                    model, tokenizer = clear_model_and_tokenizer(model, tokenizer)
                                    model = get_model(
                                        str(checkpoint_path),
                                        device=None,
                                        eval_mode=True,
                                        checkpoint_window=checkpoint_window,
                                    )
                                    tokenizer = get_tokenizer(str(checkpoint_path))

                                    current_checkpoint_window = checkpoint_window
                                    checkpoint_changed_this_window = True

                                    # Leader updates barrier so followers know checkpoint is ready
                                    if barrier.is_leader:
                                        barrier.update_checkpoint(
                                            str(checkpoint_path), checkpoint_window
                                        )
                                        # Also broadcast via Redis for faster cross-node sync
                                        if redis_aggregator is not None:
                                            redis_aggregator.broadcast_checkpoint(
                                                checkpoint_window, str(checkpoint_path)
                                            )

                                    # Log model configuration details
                                    if torch.cuda.is_available():
                                        logger.info(
                                            f"[miner] Checkpoint loaded successfully: "
                                            f"window={checkpoint_window}, "
                                            f"GPU Memory: allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GB, "
                                            f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB"
                                        )
                                        torch.cuda.empty_cache()

                                    # Start or reload vLLM server with new checkpoint
                                    if os.getenv("GRAIL_USE_VLLM", "0") == "1":
                                        gpu_mem_util = float(
                                            os.getenv("GRAIL_VLLM_GPU_MEMORY_UTIL", "0.50")
                                        )
                                        # Stagger vLLM startup to reduce contention
                                        # Workers start with 0.5-2s delay based on worker_id
                                        stagger_delay = worker_config.worker_id * 0.25
                                        if stagger_delay > 0:
                                            logger.debug(
                                                "Worker %d: waiting %.1fs before vLLM startup (stagger)",
                                                worker_config.worker_id,
                                                stagger_delay,
                                            )
                                            await asyncio.sleep(stagger_delay)

                                        if vllm_server is None:
                                            # First time - create and start server
                                            vllm_server = _create_vllm_server_manager(
                                                str(checkpoint_path),
                                                worker_config.worker_id,
                                                gpu_memory_util=gpu_mem_util,
                                            )
                                            if vllm_server is not None:
                                                logger.info(
                                                    "🚀 Starting vLLM server for worker %d...",
                                                    worker_config.worker_id,
                                                )
                                                await vllm_server.start_server()
                                                # Set URL and model name for AgentEnvLoop to use
                                                os.environ["GRAIL_VLLM_URL"] = vllm_server.base_url
                                                os.environ["GRAIL_VLLM_MODEL_NAME"] = vllm_server.model_name
                                                logger.info(
                                                    "✅ vLLM server ready at %s (model=%s)",
                                                    vllm_server.base_url,
                                                    vllm_server.model_name,
                                                )
                                                # Register cleanup on shutdown
                                                self.register_shutdown_callback(
                                                    vllm_server._stop_server
                                                )
                                        else:
                                            # Reload server with new checkpoint
                                            logger.info(
                                                "🔄 Reloading vLLM server with checkpoint window %s...",
                                                checkpoint_window,
                                            )
                                            try:
                                                await vllm_server.reload_with_new_checkpoint(
                                                    str(checkpoint_path)
                                                )
                                                # Update model name after reload
                                                os.environ["GRAIL_VLLM_MODEL_NAME"] = vllm_server.model_name
                                                logger.info(
                                                    "✅ vLLM server reloaded at %s (model=%s)",
                                                    vllm_server.base_url,
                                                    vllm_server.model_name,
                                                )
                                            except Exception as vllm_exc:
                                                logger.error(
                                                    "Failed to reload vLLM server: %s - will recreate",
                                                    vllm_exc,
                                                )
                                                # Reload failed - recreate the server from scratch
                                                try:
                                                    await vllm_server._stop_server()
                                                except Exception:
                                                    pass  # Best effort cleanup
                                                vllm_server = _create_vllm_server_manager(
                                                    str(checkpoint_path),
                                                    worker_config.worker_id,
                                                    gpu_memory_util=gpu_mem_util,
                                                )
                                                if vllm_server is not None:
                                                    logger.info(
                                                        "🔄 Recreating vLLM server for worker %d...",
                                                        worker_config.worker_id,
                                                    )
                                                    await vllm_server.start_server()
                                                    os.environ["GRAIL_VLLM_URL"] = vllm_server.base_url
                                                    os.environ["GRAIL_VLLM_MODEL_NAME"] = vllm_server.model_name
                                                    logger.info(
                                                        "✅ vLLM server recreated at %s",
                                                        vllm_server.base_url,
                                                    )
                                                    self.register_shutdown_callback(
                                                        vllm_server._stop_server
                                                    )
                                except Exception:
                                    logger.exception(
                                        "[miner] FAILED to load checkpoint for window %s from %s. "
                                        "Check logs above for hash verification or reconstruction errors.",
                                        checkpoint_window,
                                        checkpoint_path,
                                    )
                                    raise
                            else:
                                logger.warning(
                                    "[miner] Checkpoint window %s NOT AVAILABLE (get_checkpoint returned None). "
                                    "This may be due to: (1) checkpoint not published yet, "
                                    "(2) download failure, (3) hash verification failure during delta reconstruction. "
                                    "Retaining current model (window=%s)",
                                    checkpoint_window,
                                    current_checkpoint_window,
                                )
                        else:
                            # Checkpoint unchanged - hub should still broadcast so non-hub leaders
                            # know to skip R2 discovery and proceed
                            if is_hub and redis_aggregator is not None:
                                redis_aggregator.broadcast_checkpoint(
                                    current_checkpoint_window, f"unchanged-{current_checkpoint_window}"
                                )
                                logger.info(
                                    f"🌐 Hub: Checkpoint unchanged ({current_checkpoint_window}), "
                                    f"broadcast to Redis so non-hub can skip discovery"
                                )
                    elif model is None or tokenizer is None:
                        logger.error("No checkpoint available and no model loaded, cannot mine")
                        await asyncio.sleep(60)
                        continue

                    # Ensure model and tokenizer are loaded before mining
                    if model is None or tokenizer is None:
                        logger.error("Model or tokenizer not loaded, cannot mine")
                        await asyncio.sleep(30)
                        continue

                    logger.info(
                        f"🔥 Starting inference generation for window "
                        f"{window_start}-{window_start + WINDOW_LENGTH - 1}"
                    )

                    if not await has_time_for_next_generation(subtensor, timers, window_start):
                        # Only leader marks window as done when skipping due to time.
                        # Followers should NOT mark done - they need to stay in sync with leader.
                        # If follower marks done here (e.g., after waiting for checkpoint sync),
                        # they'll jump to waiting for next window while leader is still mining current.
                        if barrier.is_leader:
                            last_window_start = window_start
                        await asyncio.sleep(5)
                        continue

                    window_block_hash, combined_randomness = await get_window_randomness(
                        subtensor,
                        window_start,
                        self.use_drand,
                    )

                    # Create abort check callback for followers to detect leader downloading
                    def abort_check() -> bool:
                        return barrier.is_leader_downloading(current_checkpoint_window)

                    # Background prefetch task reference (leader only)
                    prefetch_task: asyncio.Task | None = None

                    inferences = await generate_rollouts_for_window(
                        wallet,
                        model,
                        tokenizer,
                        subtensor,
                        window_start,
                        window_block_hash,
                        combined_randomness,
                        timers,
                        monitor,
                        self.use_drand,
                        checkpoint_window,
                        worker_config,  # Multi-worker support
                        abort_check=abort_check,
                        redis_aggregator=redis_aggregator,  # For centralized block sync
                    )

                    # Note: prefetch_task persists across iterations - don't reset here
                    # It will be awaited at checkpoint loading time in the next iteration

                    if inferences:
                        # Choose aggregation mode: Redis (preferred) or file-based (fallback)
                        if redis_aggregator is not None:
                            # REDIS MODE: Rollouts already pushed incrementally during generation
                            # Just signal we're done so hub knows to aggregate
                            redis_aggregator.signal_done(window_start)

                            if redis_aggregator.is_hub:
                                # Hub: wait for workers to finish their current problem
                                # Workers check stop signal at loop top, so they may still be
                                # mid-generation when hub finishes. Give them time to push.
                                logger.info("⏳ Hub waiting 15s for workers to finish current problems...")
                                await asyncio.sleep(15)
                                # Hub: aggregate from Redis (rollouts already pushed incrementally)
                                all_inferences = redis_aggregator.aggregate_from_workers(window_start)
                                logger.info(
                                    f"🌐 Hub aggregated {len(all_inferences)} rollouts from all workers"
                                )

                                # Filter out rollouts with mismatched checkpoint_window
                                if all_inferences and current_checkpoint_window is not None:
                                    original_count = len(all_inferences)
                                    all_inferences = [
                                        inf for inf in all_inferences
                                        if inf.get("checkpoint_window") == current_checkpoint_window
                                    ]
                                    filtered_count = original_count - len(all_inferences)
                                    if filtered_count > 0:
                                        logger.warning(
                                            f"⚠️ Filtered {filtered_count}/{original_count} rollouts "
                                            f"with mismatched checkpoint_window (expected {current_checkpoint_window})"
                                        )

                                # Hub uploads
                                if all_inferences:
                                    # PREFETCH: Start background checkpoint download BEFORE upload
                                    # This allows checkpoint download to run in parallel with upload
                                    next_window = window_start + WINDOW_LENGTH

                                    async def _prefetch_next_checkpoint_redis():
                                        """Background task to prefetch checkpoint for next window."""
                                        try:
                                            next_ckpt = await checkpoint_manager.get_latest_ready_checkpoint(
                                                next_window
                                            )
                                            if next_ckpt is not None and next_ckpt != current_checkpoint_window:
                                                logger.info(
                                                    f"🔮 Prefetching checkpoint {next_ckpt} for next window {next_window}..."
                                                )
                                                ckpt_path = await checkpoint_manager.get_checkpoint(next_ckpt)
                                                if ckpt_path:
                                                    barrier.update_checkpoint(str(ckpt_path), next_ckpt)
                                                    logger.info(
                                                        f"✅ Prefetched checkpoint {next_ckpt} ready at {ckpt_path}"
                                                    )
                                        except Exception as e:
                                            logger.debug(f"Prefetch failed (non-critical): {e}")

                                    # Start prefetch in background (runs concurrently with upload)
                                    prefetch_task = asyncio.create_task(_prefetch_next_checkpoint_redis())

                                    logger.info(
                                        f"📤 Uploading {len(all_inferences)} aggregated rollouts "
                                        f"to R2 for window {window_start}..."
                                    )
                                    try:
                                        upload_duration = await upload_inferences_with_metrics(
                                            wallet, window_start, all_inferences, credentials, monitor,
                                        )
                                        timers.update_upload_time_ema(upload_duration)
                                        logger.info(
                                            f"✅ Successfully uploaded window {window_start} "
                                            f"with {len(all_inferences)} aggregated rollouts"
                                        )
                                        ts = datetime.now().strftime("%H:%M:%S")
                                        logger.info(
                                            f"[SUMMARY] W0 | window={window_start} | rollouts={len(all_inferences)} | UPLOADED | {ts}"
                                        )
                                        self.heartbeat()
                                        if monitor:
                                            await monitor.log_counter("mining/successful_uploads")
                                            await monitor.log_gauge("mining/uploaded_rollouts", len(all_inferences))
                                        # Hub cleanup Redis keys
                                        redis_aggregator.cleanup_window(window_start)
                                    except Exception as e:
                                        logger.error(f"❌ Failed to upload window {window_start}: {e}")
                                        logger.error(traceback.format_exc())
                                        if monitor:
                                            await monitor.log_counter("mining/failed_uploads")
                            else:
                                # Non-hub worker: rollouts pushed incrementally, signaled done
                                ts = datetime.now().strftime("%H:%M:%S")
                                logger.info(
                                    f"[SUMMARY] {redis_aggregator.worker_key} | window={window_start} | "
                                    f"rollouts={len(inferences)} | DONE | {ts}"
                                )

                        elif rollout_staging is not None:
                            # FILE-BASED MODE: Local staging (fallback when Redis not available)
                            if rollout_staging.is_window_uploaded(window_start):
                                logger.warning(
                                    f"⚠️ Window {window_start} already uploaded, "
                                    f"discarding {len(inferences)} late rollouts"
                                )
                            else:
                                rollout_staging.save_rollouts(window_start, inferences)

                            if barrier.is_leader:
                                # Leader: wait for other workers, aggregate, upload
                                logger.info(
                                    f"📦 Leader waiting for workers to stage rollouts..."
                                )
                                await rollout_staging.wait_for_workers(
                                    window_start, timeout=30.0
                                )
                                all_inferences = rollout_staging.aggregate_rollouts(window_start)

                                # Filter out rollouts with mismatched checkpoint_window
                                if all_inferences and current_checkpoint_window is not None:
                                    original_count = len(all_inferences)
                                    all_inferences = [
                                        inf for inf in all_inferences
                                        if inf.get("checkpoint_window") == current_checkpoint_window
                                    ]
                                    filtered_count = original_count - len(all_inferences)
                                    if filtered_count > 0:
                                        logger.warning(
                                            f"⚠️ Filtered {filtered_count}/{original_count} rollouts "
                                            f"with mismatched checkpoint_window (expected {current_checkpoint_window})"
                                        )

                                if all_inferences:
                                    # PREFETCH: Start background checkpoint download BEFORE upload
                                    # This allows checkpoint download to run in parallel with upload
                                    next_window = window_start + WINDOW_LENGTH

                                    async def _prefetch_next_checkpoint():
                                        """Background task to prefetch checkpoint for next window."""
                                        try:
                                            next_ckpt = await checkpoint_manager.get_latest_ready_checkpoint(
                                                next_window
                                            )
                                            if next_ckpt is not None and next_ckpt != current_checkpoint_window:
                                                logger.info(
                                                    f"🔮 Prefetching checkpoint {next_ckpt} for next window {next_window}..."
                                                )
                                                ckpt_path = await checkpoint_manager.get_checkpoint(next_ckpt)
                                                if ckpt_path:
                                                    barrier.update_checkpoint(str(ckpt_path), next_ckpt)
                                                    logger.info(
                                                        f"✅ Prefetched checkpoint {next_ckpt} ready at {ckpt_path}"
                                                    )
                                        except Exception as e:
                                            logger.debug(f"Prefetch failed (non-critical): {e}")

                                    # Start prefetch in background (runs concurrently with upload)
                                    prefetch_task = asyncio.create_task(_prefetch_next_checkpoint())

                                    logger.info(
                                        f"📤 Uploading {len(all_inferences)} aggregated rollouts "
                                        f"to R2 for window {window_start}..."
                                    )
                                    try:
                                        upload_duration = await upload_inferences_with_metrics(
                                            wallet, window_start, all_inferences, credentials, monitor,
                                        )
                                        timers.update_upload_time_ema(upload_duration)
                                        logger.info(
                                            f"✅ Successfully uploaded window {window_start} "
                                            f"with {len(all_inferences)} aggregated rollouts"
                                        )
                                        ts = datetime.now().strftime("%H:%M:%S")
                                        logger.info(
                                            f"[SUMMARY] W0 | window={window_start} | rollouts={len(all_inferences)} | UPLOADED | {ts}"
                                        )
                                        self.heartbeat()
                                        if monitor:
                                            await monitor.log_counter("mining/successful_uploads")
                                            await monitor.log_gauge("mining/uploaded_rollouts", len(all_inferences))
                                        rollout_staging.mark_window_uploaded(window_start)
                                    except Exception as e:
                                        logger.error(f"❌ Failed to upload window {window_start}: {e}")
                                        logger.error(traceback.format_exc())
                                        if monitor:
                                            await monitor.log_counter("mining/failed_uploads")

                                # Cleanup staging files
                                rollout_staging.cleanup_window(window_start)
                            else:
                                # Follower: staging done, skip R2 upload
                                logger.info(
                                    f"📦 Worker {worker_config.worker_id} staged "
                                    f"{len(inferences)} rollouts, leader will upload"
                                )
                                # Clean summary line for easy PM2 log monitoring
                                ts = datetime.now().strftime("%H:%M:%S")
                                logger.info(
                                    f"[SUMMARY] W{worker_config.worker_id} | window={window_start} | rollouts={len(inferences)} | STAGED | {ts}"
                                )
                                if monitor:
                                    await monitor.log_gauge("mining/staged_rollouts", len(inferences))
                        else:
                            # Single-worker mode: upload directly
                            # PREFETCH: Start background checkpoint download BEFORE upload
                            # This allows checkpoint download to run in parallel with upload
                            next_window = window_start + WINDOW_LENGTH

                            async def _prefetch_next_checkpoint_single():
                                """Background task to prefetch checkpoint for next window."""
                                try:
                                    next_ckpt = await checkpoint_manager.get_latest_ready_checkpoint(
                                        next_window
                                    )
                                    if next_ckpt is not None and next_ckpt != current_checkpoint_window:
                                        logger.info(
                                            f"🔮 Prefetching checkpoint {next_ckpt} for next window {next_window}..."
                                        )
                                        ckpt_path = await checkpoint_manager.get_checkpoint(next_ckpt)
                                        if ckpt_path:
                                            barrier.update_checkpoint(str(ckpt_path), next_ckpt)
                                            logger.info(
                                                f"✅ Prefetched checkpoint {next_ckpt} ready at {ckpt_path}"
                                            )
                                except Exception as e:
                                    logger.debug(f"Prefetch failed (non-critical): {e}")

                            # Start prefetch in background (runs concurrently with upload)
                            prefetch_task = asyncio.create_task(_prefetch_next_checkpoint_single())

                            logger.info(
                                f"📤 Uploading {len(inferences)} rollouts to R2 "
                                f"for window {window_start}..."
                            )
                            try:
                                upload_duration = await upload_inferences_with_metrics(
                                    wallet, window_start, inferences, credentials, monitor,
                                )
                                timers.update_upload_time_ema(upload_duration)
                                logger.info(
                                    f"✅ Successfully uploaded window {window_start} "
                                    f"with {len(inferences)} rollouts"
                                )
                                # Clean summary line for easy PM2 log monitoring
                                ts = datetime.now().strftime("%H:%M:%S")
                                logger.info(
                                    f"[SUMMARY] W0 | window={window_start} | rollouts={len(inferences)} | UPLOADED | {ts}"
                                )
                                self.heartbeat()
                                if monitor:
                                    await monitor.log_counter("mining/successful_uploads")
                                    await monitor.log_gauge("mining/uploaded_rollouts", len(inferences))
                                # Mark window as uploaded (single-worker mode)
                                if rollout_staging:
                                    rollout_staging.mark_window_uploaded(window_start)

                            except Exception as e:
                                logger.error(f"❌ Failed to upload window {window_start}: {e}")
                                logger.error(traceback.format_exc())
                                if monitor:
                                    await monitor.log_counter("mining/failed_uploads")
                    else:
                        logger.warning(f"No inferences generated for window {window_start}")
                        if monitor:
                            await monitor.log_counter("mining/empty_windows")

                        # Followers: if we had no rollouts (likely aborted due to checkpoint),
                        # wait for leader to finish downloading before continuing.
                        # This prevents followers from jumping ahead while leader downloads.
                        if not barrier.is_leader:
                            # Check if leader is downloading - if so, wait for it
                            if barrier.is_leader_downloading():
                                logger.info(
                                    f"⏳ Worker {worker_config.worker_id} waiting for leader "
                                    f"to finish checkpoint download..."
                                )
                                await barrier.wait_for_checkpoint_sync(
                                    current_checkpoint_window, timeout=300.0
                                )
                            # After download complete, sync to leader's window
                            leader_window = barrier.get_leader_window()
                            if leader_window is not None and leader_window != window_start:
                                logger.info(
                                    f"📍 Worker {worker_config.worker_id} syncing to leader's "
                                    f"window {leader_window} (was on {window_start})"
                                )
                                # Don't update last_window_start - let next iteration handle it
                                continue
                            # Follower with no inferences should NOT mark window as done.
                            # Let them retry the same window instead of jumping ahead.
                            # This prevents followers from waiting for next window while
                            # leader is still mining current window.
                            continue

                    last_window_start = window_start

                    # Prefetch status check - keep reference if still running
                    # Next iteration will await it with timeout at checkpoint loading time
                    if prefetch_task is not None:
                        if not prefetch_task.done():
                            logger.info(
                                "⏩ Prefetch still running - will await at next window start"
                            )
                            # DON'T set to None - keep reference for next iteration
                        else:
                            # Prefetch completed, check for errors and clear reference
                            try:
                                prefetch_task.result()
                                logger.info("✅ Prefetch completed successfully")
                            except Exception as e:
                                logger.debug(f"Prefetch completed with error (non-critical): {e}")
                            prefetch_task = None

                    # Only leader should cleanup checkpoints, and only after a new
                    # checkpoint was loaded (ensures all workers have moved to new one)
                    # DISABLED: Keep checkpoints to avoid re-downloading base for delta reconstruction
                    # if barrier.is_leader and checkpoint_changed_this_window:
                    #     await checkpoint_manager.cleanup_local(window_start)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error in miner loop: {e}. Continuing ...")
                    self.reset_subtensor()  # Force reconnect on next iteration
                    await asyncio.sleep(10)
                    continue

        # Start process-level watchdog (handled by BaseNeuron)
        self.start_watchdog(timeout_seconds=(60 * 10))
        await _run()
