#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import asyncio
import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import bittensor as bt
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..environments.factory import create_env
from ..environments.loop import AgentEnvLoop
from ..grail import derive_env_seed
from ..infrastructure.comms import sink_window_inferences
from ..infrastructure.drand import get_drand_beacon
from ..infrastructure.redis_problem_queue import get_problem_queue
from ..infrastructure.worker_barrier import WorkerBarrier
from ..infrastructure.worker_config import WorkerConfig, log_multi_gpu_setup
from ..shared.constants import (
    BLOCK_TIME_SECONDS,
    CHALLENGE_K,
    LAYER_INDEX,
    ROLLOUTS_PER_PROBLEM,
    WINDOW_LENGTH,
)
from . import console

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
logger = logging.getLogger("grail")


# --------------------------------------------------------------------------- #
#                       Styling & configuration constants                     #
# --------------------------------------------------------------------------- #
# Mining timing and safety parameters. Centralized for easy tuning and clarity.
EMA_ALPHA = 0.2  # Exponential moving average smoothing

MINER_SAFETY_BLOCKS = int(  # Safety margin blocks before window end
    os.getenv("GRAIL_MINER_SAFETY_BLOCKS", "1")
)
MINER_BUFFER_SECONDS = float(  # Extra seconds buffer for fine-tuning upload timing
    os.getenv("GRAIL_MINER_BUFFER_SECONDS", "0")
)
DEBUG_TEXT_LOG_LIMIT_PER_WINDOW = 5  # Max sample texts logged per window

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #


def get_conf(key: str, default: Any = None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        console.print(f"[red]{key} not set.[/red]\nRun:\n    af set {key} <value>")
        raise typer.Exit(code=1)
    return v or default


# --------------------------------------------------------------------------- #
#                        Helper Functions                                     #
# --------------------------------------------------------------------------- #


def parse_filename(
    filename: str,
) -> tuple[str | None, int | None, int | None]:
    """Parse filename to extract wallet, block, nonce"""
    # Remove prefix and extension
    basename = filename.split("/")[-1].replace(".json", "")
    parts = basename.split("-")
    if len(parts) >= 3:
        wallet = parts[0]
        block = int(parts[1])
        nonce = int(parts[2])
        return wallet, block, nonce
    return None, None, None


def parse_window_filename(
    filename: str,
) -> tuple[str | None, int | None]:
    """Parse window filename to extract wallet and window_start"""
    # Remove prefix and extension
    basename = filename.split("/")[-1].replace(".json", "")
    # Format: {wallet}-window-{window_start}
    parts = basename.split("-")
    if len(parts) >= 3 and parts[1] == "window":
        wallet = parts[0]
        window_start = int(parts[2])
        return wallet, window_start
    return None, None


def sign_rollout(rollout_data: dict[str, Any], wallet: bt.wallet) -> dict[str, Any]:
    """Sign a rollout using the wallet hotkey (env-agnostic)."""
    # Create challenge string from key rollout data
    episode_seed = rollout_data.get("episode_seed", rollout_data.get("sat_seed", ""))
    block_hash = rollout_data.get("block_hash", "")
    nonce = rollout_data.get("nonce", "")

    # Validate input types
    if not isinstance(block_hash, str):
        raise ValueError(f"block_hash must be str, got {type(block_hash).__name__}")
    if not isinstance(nonce, (int, str)):
        raise ValueError(f"nonce must be int or str, got {type(nonce).__name__}")

    # Use delimiter to prevent collision attacks
    challenge = f"{episode_seed}|{block_hash}|{nonce}"
    rollout_data["challenge"] = challenge
    rollout_data["hotkey"] = wallet.hotkey.ss58_address
    # Encode challenge to bytes before signing (explicit UTF-8)
    signature = wallet.hotkey.sign(data=challenge.encode("utf-8")).hex()
    rollout_data["signature"] = signature
    return rollout_data


# --------------------------------------------------------------------------- #
#                         Time & window utilities                             #
# --------------------------------------------------------------------------- #


def calculate_window_start(block_number: int) -> int:
    return (block_number // WINDOW_LENGTH) * WINDOW_LENGTH


@dataclass
class MiningTimers:
    """Tracks time estimates and exponential moving averages (EMAs).

    We keep EMAs of block time, generation time, and upload time to make
    conservative, adaptive decisions about whether there's enough time left
    in the current window to safely generate and upload another batch.
    """

    block_time_ema_s: float = float(BLOCK_TIME_SECONDS)
    gen_time_ema_s: float | None = None
    upload_time_ema_s: float | None = None
    last_block_num: int | None = None
    last_block_ts: float | None = None

    def update_block_time_ema(self, current_block: int) -> None:
        """Update the EMA for block time using observed block deltas.

        Uses the time elapsed between the last seen block and the current block
        to update an EMA of the chain's average block time.
        """
        now_ts = time.time()
        if self.last_block_num is not None and self.last_block_ts is not None:
            dn = current_block - self.last_block_num
            if dn > 0:
                dt = now_ts - self.last_block_ts
                if dt > 0.0:
                    sample_bt = dt / dn
                    self.block_time_ema_s = (
                        EMA_ALPHA * sample_bt + (1.0 - EMA_ALPHA) * self.block_time_ema_s
                    )
        self.last_block_num = current_block
        self.last_block_ts = now_ts

    def blocks_needed_for_next_gen(self) -> int:
        """Estimate how many blocks we need to finish a gen+upload safely.

        Combines gen time EMA, upload time EMA, and a safety margin (in blocks)
        to convert projected seconds into blocks remaining in the window.
        """
        # Default gen time: 3 blocks (previously 6, which was too conservative)
        est_gen_s = (
            self.gen_time_ema_s if self.gen_time_ema_s is not None else 3.0 * self.block_time_ema_s
        )
        est_upload_s = (
            self.upload_time_ema_s
            if self.upload_time_ema_s is not None
            else 1.0 * self.block_time_ema_s
        )
        safety_s = float(MINER_SAFETY_BLOCKS) * self.block_time_ema_s
        total_s = est_gen_s + est_upload_s + safety_s + MINER_BUFFER_SECONDS
        # Cap at MINER_SAFETY_BLOCKS to avoid stopping too early due to inflated gen_time_ema
        return min(MINER_SAFETY_BLOCKS, max(1, math.ceil(total_s / max(0.001, self.block_time_ema_s))))

    def update_gen_time_ema(self, duration_s: float) -> None:
        # Cap first generation time to avoid model warmup skewing the EMA
        # First gens can take 50-60s due to CUDA/vLLM compilation, but steady state is ~25-30s
        MAX_FIRST_GEN_S = 40.0
        if self.gen_time_ema_s is None:
            self.gen_time_ema_s = min(duration_s, MAX_FIRST_GEN_S)
        else:
            self.gen_time_ema_s = EMA_ALPHA * duration_s + (1.0 - EMA_ALPHA) * self.gen_time_ema_s

    def update_upload_time_ema(self, duration_s: float) -> None:
        self.upload_time_ema_s = (
            duration_s
            if self.upload_time_ema_s is None
            else EMA_ALPHA * duration_s + (1.0 - EMA_ALPHA) * self.upload_time_ema_s
        )


async def has_time_for_next_generation(
    subtensor: bt.subtensor, timers: MiningTimers, window_start: int
) -> bool:
    """Return True if there is enough time left to run one more gen+upload.

    Args:
        subtensor: Bittensor subtensor client for chain reads.
        timers: Moving averages and block-time state.
        window_start: Start block number of the current window.

    Returns:
        True if blocks remaining > conservative estimate of blocks required.
    """
    current_check = await subtensor.get_current_block()
    timers.update_block_time_ema(current_check)
    blocks_remaining = (window_start + WINDOW_LENGTH) - current_check
    needed_blocks = timers.blocks_needed_for_next_gen()
    if blocks_remaining <= needed_blocks:
        logger.warning(
            "Window %s nearly over (block %s); need %s blocks to safely "
            "finish next generation+upload.",
            window_start,
            current_check,
            needed_blocks,
        )
        return False
    return True


async def get_window_randomness(
    subtensor: bt.subtensor, window_start: int, use_drand: bool
) -> tuple[str, str]:
    """Compute randomness for the window using block hash and optional drand.

    We prefer mixing the window's block hash with the drand beacon when
    available to avoid miner-controlled randomness. Falls back to block hash.

    Returns:
        (window_block_hash, combined_randomness)
    """
    window_block_hash = await subtensor.get_block_hash(window_start)
    if not use_drand:
        return window_block_hash, window_block_hash

    try:
        # Run drand HTTP request in thread pool to avoid blocking event loop
        drand_beacon = await asyncio.to_thread(get_drand_beacon, None)
        logger.info("🎲 Using drand randomness from round %s", drand_beacon["round"])
        combined_randomness = hashlib.sha256(
            (window_block_hash + drand_beacon["randomness"]).encode()
        ).hexdigest()
        return window_block_hash, combined_randomness
    except Exception as e:
        logger.warning("Failed to get drand, using block hash only: %s", e)
        return window_block_hash, window_block_hash


async def maybe_log_debug_sample(
    tokenizer: AutoTokenizer,
    sample: Any,
    window_start: int,
    base_nonce: int,
    monitor: Any | None,
    text_logs_emitted: int,
    text_log_limit: int,
) -> int:
    """Emit a single decoded sample for debugging, rate-limited per window.

    Args:
        tokenizer: Tokenizer for decoding tokens to text
        sample: Rollout sample to log
        window_start: Window start block
        base_nonce: Base nonce for the rollout group
        monitor: Optional monitoring client
        text_logs_emitted: Current count of emitted logs
        text_log_limit: Maximum logs to emit

    Returns:
        Updated text_logs_emitted counter
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return text_logs_emitted
    if text_logs_emitted >= text_log_limit:
        return text_logs_emitted
    if not sample:
        logger.debug("[TEXT_LOG] No sample to log")
        return text_logs_emitted

    # Debug: confirm we're attempting to log
    logger.debug("[TEXT_LOG] Attempting to decode sample (emitted=%d/%d)", text_logs_emitted, text_log_limit)

    try:
        prompt_len = int(getattr(sample, "prompt_length", 0) or 0)
        completion_len = int(getattr(sample, "completion_length", 0) or 0)
        sample_text = tokenizer.decode(sample.tokens, skip_special_tokens=False)
        sample_nonce = base_nonce * 10
        logger.debug(
            (
                "TEXT[mine] window=%s group=%s nonce=%s reward=%.3f "
                "adv=%.3f success=%s text=%s prompt_len=%d completion_len=%d"
            ),
            window_start,
            base_nonce,
            sample_nonce,
            float(sample.reward),
            float(sample.advantage),
            bool(sample.success),
            sample_text,
            prompt_len,
            completion_len,
        )
        if monitor:
            await monitor.log_artifact(
                "mining/sample_text",
                {
                    "window": window_start,
                    "group": base_nonce,
                    "nonce": sample_nonce,
                    "reward": float(sample.reward),
                    "advantage": float(sample.advantage),
                    "success": bool(sample.success),
                    "text": sample_text,
                },
                "text",
            )
        return text_logs_emitted + 1
    except Exception as e:
        logger.warning("[TEXT_LOG] Failed to decode sample: %s", e)
        return text_logs_emitted


def extract_assignment_from_rollout(rollout: Any) -> list[bool]:
    """Extract boolean assignment from rollout trajectory if available."""
    if rollout.trajectory and isinstance(rollout.trajectory[0][1], list):
        return rollout.trajectory[0][1]
    return []


def count_satisfied_clauses(sat_problem: Any, assignment: list[bool]) -> int:
    """Count how many SAT clauses are satisfied by a boolean assignment."""
    if not assignment:
        return 0
    satisfied = 0
    for clause in sat_problem.clauses:
        clause_satisfied = False
        for lit in clause:
            idx = abs(lit) - 1
            if idx < 0 or idx >= len(assignment):
                continue
            value = assignment[idx]
            if (lit > 0 and value) or (lit < 0 and not value):
                clause_satisfied = True
                break
        if clause_satisfied:
            satisfied += 1
    return satisfied


async def log_generation_timing(
    subtensor: bt.subtensor,
    timers: MiningTimers,
    window_start: int,
    generation_duration: float,
    rollout_count: int,
    monitor: Any | None,
    cached_block: int | None = None,
) -> bool:
    """Log generation timing metrics and check if generation finished safely.

    Args:
        subtensor: Bittensor subtensor client for block queries
        timers: Mining timers with EMA estimates
        window_start: Start block of current window
        generation_duration: Time taken for rollout generation
        rollout_count: Number of rollouts generated
        monitor: Optional monitoring client
        cached_block: Optional cached block number (avoids RPC call if provided)

    Returns:
        True if generation finished with safe buffer for upload, False otherwise
    """
    # Use cached block if provided, otherwise fetch from chain
    post_gen_block = cached_block if cached_block is not None else await subtensor.get_current_block()
    blocks_remaining = (window_start + WINDOW_LENGTH) - post_gen_block
    time_remaining_s = blocks_remaining * timers.block_time_ema_s
    needed_blocks_for_upload = max(
        MINER_SAFETY_BLOCKS, math.ceil((timers.upload_time_ema_s or 0.0) / max(0.001, timers.block_time_ema_s))
    )

    generation_safe = blocks_remaining > needed_blocks_for_upload

    logger.info(
        "Generation timing: %d blocks remaining, %.1fs left, upload needs %d blocks - %s",
        blocks_remaining,
        time_remaining_s,
        needed_blocks_for_upload,
        "✅ SAFE" if generation_safe else "⚠️ TIGHT",
    )

    if monitor:
        await monitor.log_gauge(
            "profiling/generation_finished_safely",
            1.0 if generation_safe else 0.0,
        )

    return generation_safe


def package_rollout_data(
    model: AutoModelForCausalLM,
    wallet: bt.wallet,
    rollout: Any,
    base_nonce: int,
    rollout_idx: int,
    total_in_group: int,
    window_start: int,
    current_block: int,
    window_block_hash: str,
    combined_randomness: str,
    use_drand: bool,
    checkpoint_window: int,
) -> dict:
    """Assemble the full on-chain/off-chain payload for a single rollout.

    This binds model outputs (tokens, commitments) to the randomness, model name,
    and layer via a commit-binding signature, and includes proof metadata
    required by validators.

    Args:
        model: Loaded model (for name_or_path)
        wallet: Miner wallet for signing
        rollout: Generated rollout with tokens/commitments/trajectory
        base_nonce: Base nonce for the group
        rollout_idx: Index within the group
        total_in_group: Total rollouts in group
        window_start: Window start block
        current_block: Current block
        window_block_hash: Window block hash
        combined_randomness: Challenge randomness
        use_drand: Whether drand was used
        checkpoint_window: The checkpoint window used for this rollout

    Returns:
        Signed dictionary ready to upload for validation
    """
    # Multiplier must be >= max batch size to avoid nonce collisions
    # With batch_size=16, indices go 0-15, so multiplier must be at least 16
    # Using 100 for safety margin and future batch size increases
    rollout_nonce = base_nonce * 100 + rollout_idx

    # Sign commit binding (tokens, randomness, model, layer, commitments)
    from ..protocol.signatures import sign_commit_binding

    logger.debug("Signing commit binding for rollout %s", rollout_idx)
    commit_sig = sign_commit_binding(
        tokens=rollout.tokens,
        randomness_hex=combined_randomness,
        model_name=model.name_or_path,
        layer_index=LAYER_INDEX,
        commitments=rollout.commitments,
        wallet=wallet,
    )

    assignment = extract_assignment_from_rollout(rollout)
    # satisfied_clauses retained for backward-compat field; set to 0 in env-agnostic mode
    satisfied_clauses = 0

    payload = {
        "window_start": window_start,
        "block": current_block,
        "nonce": rollout_nonce,
        "block_hash": window_block_hash,
        "randomness": combined_randomness,
        "use_drand": use_drand,
        "rollout_group": base_nonce,
        "rollout_index": rollout_idx,
        "total_in_group": total_in_group,
        "checkpoint_window": checkpoint_window,  # Explicit checkpoint used
        "commit": {
            "tokens": rollout.tokens,
            "commitments": rollout.commitments,
            "proof_version": rollout.proof_version,
            "model": {
                "name": model.name_or_path,
                "layer_index": LAYER_INDEX,
            },
            "signature": commit_sig.hex(),
            "beacon": rollout.beacon,
            "rollout": {
                "trajectory": rollout.trajectory,
                "total_reward": rollout.reward,
                "advantage": rollout.advantage,
                "success": rollout.success,
                "token_logprobs": rollout.token_logprobs,
                "prompt_length": rollout.prompt_length,
                "completion_length": rollout.completion_length,
                "satisfied_clauses": satisfied_clauses,
                "assignment": assignment,
            },
        },
        "timestamp": time.time(),
    }

    return sign_rollout(payload, wallet)


async def upload_inferences_with_metrics(
    wallet: bt.wallet,
    window_start: int,
    inferences: list[dict],
    credentials: Any,
    monitor: Any | None,
) -> float:
    """Upload window payload to object storage and return elapsed seconds.

    Args:
        wallet: Miner wallet for authentication.
        window_start: Start block of the window being uploaded.
        inferences: List of rollout data to upload.
        credentials: Object storage credentials.
        monitor: Optional monitoring client for timing metrics.

    Returns:
        Upload duration in seconds.
    """
    upload_start = time.time()
    if monitor:
        with monitor.timer("profiling/upload"):
            await sink_window_inferences(
                wallet,
                window_start,
                inferences,
                credentials,
            )
    else:
        await sink_window_inferences(
            wallet,
            window_start,
            inferences,
            credentials,
        )
    return time.time() - upload_start


async def generate_rollouts_for_window(
    wallet: bt.wallet,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    subtensor: bt.subtensor,
    window_start: int,
    window_block_hash: str,
    combined_randomness: str,
    timers: MiningTimers,
    monitor: Any | None,
    use_drand: bool,
    checkpoint_window: int,
    worker_config: WorkerConfig | None = None,
    abort_check: Callable[[], bool] | None = None,
    redis_aggregator: Any | None = None,
) -> list[dict]:
    """Generate as many GRPO rollouts as safely possible within a window.

    Core loop responsibilities:
      - Respect time budget using EMAs (stop before window end)
      - Periodically clear CUDA cache to reduce fragmentation
      - Track and log per-window metrics
      - Package each rollout with commit-binding signatures and proofs
      - Support multi-worker partitioning for parallel mining

    Args:
        wallet: Miner wallet for signing and authentication.
        model: Loaded model instance.
        tokenizer: Loaded tokenizer instance.
        subtensor: Bittensor client for chain reads.
        window_start: Start block of the current window.
        window_block_hash: Block hash at window start.
        combined_randomness: Per-window randomness for challenges.
        timers: EMA-based timing estimates for safety.
        monitor: Optional monitoring client for metrics.
        use_drand: Whether drand was used in randomness generation.
        checkpoint_window: The checkpoint window used for this generation
        worker_config: Optional multi-worker configuration for problem partitioning
        abort_check: Optional callback that returns True if generation should abort
                     (e.g., leader is downloading a new checkpoint)
        redis_aggregator: Optional Redis aggregator for centralized block sync.
                          Hub caches blocks to Redis, all workers read from Redis.

    Returns:
        List of signed rollout data ready for upload.
    """
    # Load worker config if not provided
    if worker_config is None:
        worker_config = WorkerConfig.from_env()

    # Window generation state and metrics
    inferences: list[dict] = []
    start_time = time.time()
    inference_count = 0  # Total number of problems attempted in this window
    successful_rollouts = 0
    failed_rollouts = 0
    total_reward = 0.0
    # Avoid flooding logs in debug mode
    text_logs_emitted = 0  # Running count of emitted debug texts
    worker_problem_count = 0  # Problems this worker has processed

    # Initialize shared problem queue for gap-free distribution
    # Workers atomically claim problems instead of round-robin, ensuring no gaps
    # Uses Redis if GRAIL_REDIS_URL is set (for cross-server coordination)
    cache_root = worker_config.cache_root if hasattr(worker_config, 'cache_root') else os.path.expanduser("~/.cache/grail")
    problem_queue = get_problem_queue(cache_root, worker_config.worker_id, worker_config.total_workers)

    # Initialize barrier for block sharing (leader shares block, followers read)
    barrier = WorkerBarrier(cache_root, worker_config.worker_id, worker_config.total_workers)
    is_leader = worker_config.worker_id == 0

    # Determine who should reset the problem queue counter:
    # - Multi-node mode: ONLY the hub should reset (global leader across all nodes)
    # - Single-node mode: The local leader (worker 0) should reset
    # Previously, all node leaders were resetting, causing duplicate problem indices!
    is_hub = redis_aggregator is not None and redis_aggregator.is_hub
    should_reset_counter = is_hub or (redis_aggregator is None and is_leader)

    # Track which nodes have been signaled early stop (avoid duplicate signals)
    nodes_signaled_early_stop: set[str] = set()

    if should_reset_counter:
        problem_queue.reset_counter(window_start)
    elif not is_hub and redis_aggregator is not None:
        # Non-hub nodes wait for hub to reset (even their worker-0)
        problem_queue.wait_for_leader_reset(window_start, timeout=10.0)
    else:
        # Single-node followers wait for leader reset
        problem_queue.wait_for_leader_reset(window_start, timeout=10.0)

    # Detect device from model (handles multi-GPU)
    device = getattr(model, "grail_primary_device", None) or model.device
    # Batch size for parallel rollout generation (tune per node for memory/throughput)
    batch_size = int(os.getenv("GRAIL_GENERATION_BATCH_SIZE", "2"))
    if batch_size > ROLLOUTS_PER_PROBLEM:
        logger.warning(
            "GRAIL_GENERATION_BATCH_SIZE=%d exceeds ROLLOUTS_PER_PROBLEM=%d; capping at %d",
            batch_size,
            ROLLOUTS_PER_PROBLEM,
            ROLLOUTS_PER_PROBLEM,
        )
        batch_size = ROLLOUTS_PER_PROBLEM
    loop = AgentEnvLoop(model, tokenizer, device)
    if batch_size > 1:
        logger.info("Using batch_size=%d for parallel rollout generation", batch_size)

    while True:
        # Check if we should abort generation (e.g., leader downloading new checkpoint)
        if abort_check is not None and abort_check():
            logger.warning(
                "⏸️ Aborting generation: leader is downloading a new checkpoint "
                "(discarding %d rollouts generated with old checkpoint)",
                len(inferences),
            )
            # Return empty - rollouts with old checkpoint would be filtered anyway
            return []

        # Atomically claim next problem from shared queue (gap-free distribution)
        # This replaces round-robin and ensures contiguous problem indices
        problem_index = problem_queue.claim_next_problem(window_start)
        if problem_index < 0:
            logger.error("Failed to claim problem from queue, stopping generation")
            break

        # Fetch current block to check timing
        # Hub fetches from chain and caches to Redis; all others read from Redis
        is_hub = redis_aggregator is not None and redis_aggregator.is_hub

        if is_hub:
            # Hub queries blockchain and caches to Redis
            current_block = await subtensor.get_current_block()
            redis_aggregator.cache_current_block(current_block)
            # Log periodically to show hub is updating (every 5 blocks in gen loop)
            if current_block % 5 == 0:
                logger.info("🌐 Hub gen-loop: block %d → Redis", current_block)
        elif redis_aggregator is not None:
            # All workers read from Redis only (never RPC) - ensures perfect sync
            current_block = redis_aggregator.get_cached_block()
            wait_count = 0
            while current_block is None:
                wait_count += 1
                if wait_count % 5 == 0:
                    logger.warning(
                        "⚠️ Waiting for Redis block cache in gen loop (%ds)...",
                        wait_count,
                    )
                await asyncio.sleep(1)
                current_block = redis_aggregator.get_cached_block()
            # Log successful Redis reads to confirm sync is working
            if wait_count > 0:
                logger.info("✅ Gen-loop got block %d from Redis after %ds", current_block, wait_count)
            elif current_block % 5 == 0:
                logger.info("📡 Gen-loop Redis block: %d", current_block)
        elif is_leader:
            # No Redis - local leader fetches and shares via file
            current_block = await subtensor.get_current_block()
            barrier.update_shared_block(current_block)
        else:
            # No Redis - followers read from file-based cache
            current_block = barrier.get_shared_block()
            if current_block is None:
                current_block = await subtensor.get_current_block()
        timers.update_block_time_ema(current_block)
        current_window = calculate_window_start(current_block)
        if current_window > window_start:
            logger.info("Window %s has ended, moving to next window", window_start)
            # HUB: Signal all other workers to stop (window ended)
            if is_hub and redis_aggregator is not None:
                redis_aggregator.signal_generation_stop(window_start)
            break

        # NON-HUB WORKERS: ONLY follow hub's stop signal, ignore own block timing
        # This ensures all workers stop exactly when hub decides
        if redis_aggregator is not None and not is_hub:
            if redis_aggregator.should_stop_generation(window_start):
                logger.info(
                    "🛑 Hub signaled STOP - stopping generation for window %s",
                    window_start,
                )
                break
            # Non-hub workers skip block timing check - only hub decides when to stop
        else:
            # HUB and single-node mode: Use block timing to decide when to stop
            blocks_remaining = (window_start + WINDOW_LENGTH) - current_block
            needed_blocks = timers.blocks_needed_for_next_gen()

            # HUB: Check if any nodes need early stop signal (per-node buffers)
            # This signals slow nodes to stop before the hub would normally stop.
            # E.g., GRAIL_STOP_BUFFER_node-2=2 means node-2 stops 2 blocks earlier.
            if is_hub and redis_aggregator is not None:
                nodes_to_signal = redis_aggregator.get_nodes_needing_early_stop(
                    blocks_remaining, needed_blocks
                )
                for node_id in nodes_to_signal:
                    if node_id not in nodes_signaled_early_stop:
                        redis_aggregator.signal_generation_stop_for_node(window_start, node_id)
                        nodes_signaled_early_stop.add(node_id)

            if blocks_remaining <= needed_blocks:
                logger.info(
                    (
                        "Stopping generation: %s blocks remain, need %s "
                        "(gen≈%.1fs, upload≈%.1fs, block≈%.2fs)"
                    ),
                    blocks_remaining,
                    needed_blocks,
                    (timers.gen_time_ema_s or 0.0),
                    (timers.upload_time_ema_s or 0.0),
                    timers.block_time_ema_s,
                )
                # HUB: Signal all other workers to stop
                if is_hub and redis_aggregator is not None:
                    redis_aggregator.signal_generation_stop(window_start)
                break

        try:
            gen_start = time.time()

            worker_problem_count += 1
            inference_count += 1

            # Log with worker info if in multi-worker mode
            if worker_config.total_workers > 1:
                logger.info(
                    "⚡ Worker %d/%d: Generating GRPO rollouts for problem %d (local #%d, block %s/%s)...",
                    worker_config.worker_id + 1,
                    worker_config.total_workers,
                    problem_index,
                    worker_problem_count,
                    current_block,
                    window_start + WINDOW_LENGTH - 1,
                )
            else:
                logger.info(
                    "⚡ Generating GRPO rollouts for problem %s (block %s/%s)...",
                    problem_index,
                    current_block,
                    window_start + WINDOW_LENGTH - 1,
                )

            # Periodically reclaim free memory — helpful for long runs
            if worker_problem_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(
                    "GPU memory allocated: %s MB",
                    f"{torch.cuda.memory_allocated() / 1024**2:.2f}",
                )

            # Deterministically derive environment seed from miner+window+index
            # NOTE: problem_index is used (not worker_problem_count) to ensure
            # all workers generate unique, deterministic problems
            seed_int = derive_env_seed(wallet.hotkey.ss58_address, window_block_hash, problem_index)
            # Use deterministic problem index as rollout_group identifier
            base_nonce = problem_index
            logger.debug(
                ("MINER SEED DERIVATION: hotkey=%s window_hash=%s problem_index=%d -> seed=%d"),
                wallet.hotkey.ss58_address[:12],
                window_block_hash[:12],
                problem_index,
                seed_int,
            )

            # Generate GRPO rollouts using AgentEnvLoop
            # Factory uses cached task source automatically (no manual instantiation needed)
            def _env_factory():
                return create_env()

            # Time the rollout generation for both logging and monitoring
            rollout_gen_start = time.time()
            if monitor:
                with monitor.timer("profiling/rollout_generation"):
                    grpo_rollouts = await asyncio.to_thread(
                        loop.run_grpo_group,
                        _env_factory,
                        ROLLOUTS_PER_PROBLEM,
                        combined_randomness,
                        wallet,
                        batch_size=batch_size,
                        seed=seed_int,
                    )
            else:
                grpo_rollouts = await asyncio.to_thread(
                    loop.run_grpo_group,
                    _env_factory,
                    ROLLOUTS_PER_PROBLEM,
                    combined_randomness,
                    wallet,
                    batch_size=batch_size,
                    seed=seed_int,
                )
            rollout_gen_duration = time.time() - rollout_gen_start

            if grpo_rollouts:
                text_logs_emitted = await maybe_log_debug_sample(
                    tokenizer,
                    grpo_rollouts[0],
                    window_start,
                    base_nonce,
                    monitor,
                    text_logs_emitted,
                    DEBUG_TEXT_LOG_LIMIT_PER_WINDOW,
                )

            successful_count = sum(1 for r in grpo_rollouts if r.success)
            mean_reward = (
                sum(r.reward for r in grpo_rollouts) / len(grpo_rollouts) if grpo_rollouts else 0
            )
            logger.info(
                "GRPO batch: %s/%s successful, mean reward: %.3f, generation time: %.2fs",
                successful_count,
                len(grpo_rollouts),
                mean_reward,
                rollout_gen_duration,
            )

            # Check generation timing and log metrics (use cached block to avoid RPC)
            await log_generation_timing(
                subtensor, timers, window_start, rollout_gen_duration, len(grpo_rollouts), monitor,
                cached_block=current_block,
            )

            if worker_problem_count % 2 == 0:
                elapsed = time.time() - start_time
                rollouts_per_sec = (len(inferences) / elapsed) if elapsed > 0 else 0
                logger.info(
                    ("📊 Progress: %s rollouts from %s problems in %.1fs (%.1f rollouts/sec)"),
                    len(inferences),
                    worker_problem_count,
                    elapsed,
                    rollouts_per_sec,
                )
                if monitor:
                    await monitor.log_gauge("mining/rollouts_generated", len(inferences))
                    await monitor.log_gauge("mining/problems_processed", worker_problem_count)
                    await monitor.log_gauge("mining/rollouts_per_second", rollouts_per_sec)
                    if successful_rollouts + failed_rollouts > 0:
                        success_rate = successful_rollouts / (successful_rollouts + failed_rollouts)
                        await monitor.log_gauge("mining/success_rate", success_rate)

            # ─────────────────────────────────────────────────────────────────────
            # COMPLETION LENGTH GATE: Drop entire group if any rollout is too short
            # ─────────────────────────────────────────────────────────────────────
            # Validators require at least CHALLENGE_K tokens in the completion region
            # to perform cryptographic verification (sketch checks at k=16 positions).
            # If any rollout in the group has completion_length < CHALLENGE_K, the
            # validator will reject that rollout. Rather than waste bandwidth uploading
            # a partially valid group, we drop the entire group preemptively.
            short_rollouts = [
                (i, r.completion_length)
                for i, r in enumerate(grpo_rollouts)
                if r.completion_length < CHALLENGE_K
            ]
            if short_rollouts:
                short_details = ", ".join(f"idx={i}:len={length}" for i, length in short_rollouts)
                logger.warning(
                    "Dropping group %d: %d/%d rollouts have completion < %d tokens (%s)",
                    base_nonce,
                    len(short_rollouts),
                    len(grpo_rollouts),
                    CHALLENGE_K,
                    short_details,
                )
                # Check if vLLM server is completely down (all rollouts failed)
                # If so, abort generation early to save time
                if len(short_rollouts) == len(grpo_rollouts):
                    # Check if loop's backend is unhealthy (after consecutive failures)
                    backend = getattr(loop, "_backend", None)
                    if backend is not None and hasattr(backend, "is_healthy"):
                        if not backend.is_healthy:
                            logger.error(
                                "🚨 vLLM server appears DOWN - aborting generation early "
                                "(consecutive failed batches detected)"
                            )
                            break  # Exit generation loop, push what we have
                # Skip packaging this group entirely; continue to next problem
                timers.update_gen_time_ema(time.time() - gen_start)
                continue

            # Package each rollout with signatures and proofs for validation
            for rollout_idx, rollout in enumerate(grpo_rollouts):
                rollout_data = package_rollout_data(
                    model,
                    wallet,
                    rollout,
                    base_nonce,
                    rollout_idx,
                    len(grpo_rollouts),
                    window_start,
                    current_block,
                    window_block_hash,
                    combined_randomness,
                    use_drand,
                    checkpoint_window,
                )
                inferences.append(rollout_data)

                if rollout.success:
                    successful_rollouts += 1
                    total_reward += rollout.reward
                    if monitor:
                        await monitor.log_counter("mining/successful_rollouts")
                        await monitor.log_histogram("mining/reward_distribution", rollout.reward)
                else:
                    failed_rollouts += 1
                    if monitor:
                        await monitor.log_counter("mining/failed_rollouts")

            timers.update_gen_time_ema(time.time() - gen_start)
            await asyncio.sleep(0.01)

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error("CUDA error at inference %s: %s", inference_count, e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            logger.warning("Failed to generate inference %s: %s", inference_count, e)
            continue

    elapsed_time = time.time() - start_time
    avg_gen_time = timers.gen_time_ema_s or 0.0

    logger.info(
        "🎯 Generated %s rollouts in %.1fs for window %s (avg gen time: %.2fs/problem)",
        len(inferences),
        elapsed_time,
        window_start,
        avg_gen_time,
    )
    if monitor:
        await monitor.log_counter("mining/windows_completed")
        await monitor.log_gauge(
            "profiling/window_duration",
            elapsed_time,
        )
        await monitor.log_gauge("mining/total_rollouts_in_window", len(inferences))
        await monitor.log_gauge(
            "profiling/average_generation_time",
            avg_gen_time,
        )
        if successful_rollouts + failed_rollouts > 0:
            final_success_rate = successful_rollouts / (successful_rollouts + failed_rollouts)
            await monitor.log_gauge("mining/final_success_rate", final_success_rate)
        if successful_rollouts > 0:
            avg_reward = total_reward / successful_rollouts
            await monitor.log_gauge("mining/average_reward", avg_reward)

    return inferences


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
def register(app: typer.Typer) -> None:
    app.command("mine")(mine)


# (Watchdog removed; handled by BaseNeuron in MinerNeuron)


# --------------------------------------------------------------------------- #
#                               MINER                                         #
# --------------------------------------------------------------------------- #
def mine(
    use_drand: bool = typer.Option(
        True,
        "--use-drand/--no-drand",
        help="Use drand for randomness (default: True)",
        show_default=True,
    ),
) -> None:
    """Mine GRPO rollouts for SAT problems using GRAIL proofs.

    Stage 2: delegate to MinerNeuron lifecycle to keep behavior identical
    while standardizing the long-running process management.
    """
    from ..neurons import MinerNeuron

    asyncio.run(MinerNeuron(use_drand=use_drand).main())


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    mine()
