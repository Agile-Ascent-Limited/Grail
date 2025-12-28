from __future__ import annotations

import asyncio
import contextlib
import logging
import os
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
from grail.infrastructure.chain import GrailChainManager
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

                # Get subtensor and metagraph for chain manager
                subtensor = await self.get_subtensor()
                self.heartbeat()
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

            else:
                # FOLLOWER (workers 1-7): Wait for leader, skip blockchain init
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

            while not self.stop_event.is_set():
                try:
                    # Track if checkpoint changed this iteration (for cleanup timing)
                    checkpoint_changed_this_window = False

                    # Update heartbeat at start of each iteration
                    self.heartbeat()

                    # Use shared subtensor from base class
                    subtensor = await self.get_subtensor()

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
                            )

                        await asyncio.sleep(2)
                        continue

                    # Window is available - reset tracker
                    window_wait_tracker.reset()

                    # Leader signals which window it's working on so followers can sync
                    if barrier.is_leader:
                        barrier.signal_current_window(window_start)
                        # Also signal via Redis for cross-node sync
                        if redis_aggregator is not None:
                            redis_aggregator.signal_window_start(window_start)

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

                    # Discover latest ready checkpoint (before current window)
                    # This allows miners to proceed even if trainer is lagging
                    checkpoint_window = await checkpoint_manager.get_latest_ready_checkpoint(
                        window_start
                    )

                    # Load checkpoint if discovered and different from current
                    if checkpoint_window is not None and checkpoint_window >= 0:
                        if current_checkpoint_window != checkpoint_window:
                            # Leader signals to followers that it's downloading
                            if barrier.is_leader:
                                barrier.signal_checkpoint_downloading(checkpoint_window)

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
                                                    "Failed to reload vLLM server: %s",
                                                    vllm_exc,
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
                    )

                    # Prefetch task will be started AFTER upload completes (see below)
                    prefetch_task = None

                    if inferences:
                        # Choose aggregation mode: Redis (preferred) or file-based (fallback)
                        if redis_aggregator is not None:
                            # REDIS MODE: Each worker pushes directly to Redis
                            # No file staging needed - Redis handles everything
                            logger.info(
                                f"🌐 Worker {redis_aggregator.worker_key} pushing {len(inferences)} rollouts to Redis..."
                            )
                            redis_aggregator.push_rollouts(window_start, inferences)

                            if redis_aggregator.is_hub:
                                # Hub: wait for all workers and aggregate
                                expected_workers = redis_aggregator.total_nodes * redis_aggregator.total_workers
                                logger.info(
                                    f"🌐 Hub waiting for {expected_workers} workers..."
                                )
                                await redis_aggregator.wait_for_workers(
                                    window_start, timeout=60.0
                                )
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
                                # Non-hub worker: just pushed to Redis, done
                                ts = datetime.now().strftime("%H:%M:%S")
                                logger.info(
                                    f"[SUMMARY] {redis_aggregator.worker_key} | window={window_start} | "
                                    f"rollouts={len(inferences)} | PUSHED | {ts}"
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

                    # Wait for prefetch task to complete before next window
                    # This ensures the checkpoint is fully downloaded/reconstructed
                    if prefetch_task is not None:
                        try:
                            await prefetch_task
                        except Exception as e:
                            logger.debug(f"Prefetch task error (non-critical): {e}")
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
