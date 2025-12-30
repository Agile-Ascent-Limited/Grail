"""
Worker barrier for leader-follower synchronization in multi-GPU mining.

Worker 0 (leader) does initialization (blockchain, checkpoint download) and signals ready.
Workers 1-7 (followers) wait for leader, then proceed with mining only.

Also provides ProblemQueue for gap-free problem distribution across workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Barrier file location (relative to cache root)
BARRIER_DIR = ".worker-barrier"
BARRIER_FILE = "leader-ready.json"
RUN_ID_FILE = "run-id.txt"
BLOCK_CACHE_FILE = "current-block.txt"  # Lightweight block sharing

# How long barrier data is considered fresh (seconds)
BARRIER_MAX_AGE = 600  # 10 minutes

# Barrier files older than this are definitely stale and should be deleted
# (even by followers) - prevents 25-hour-old files from blocking startup
BARRIER_STALE_THRESHOLD = 3600  # 1 hour - anything older is definitely stale

# Maximum problems per window (configurable via env var)
# Default 500 supports ~8 H200s at 2 rollouts/s for 300s windows
MAX_PROBLEMS_PER_WINDOW = int(os.getenv("GRAIL_MAX_PROBLEMS_PER_WINDOW", "500"))
BLOCK_CACHE_MAX_AGE = 30  # Block cache stale after 30 seconds


class WorkerBarrier:
    """
    Coordinates worker initialization using a file-based barrier.

    Leader (worker 0):
        - Does all blockchain initialization
        - Downloads checkpoints
        - Writes barrier file when ready
        - Cleans up barrier on shutdown

    Followers (workers 1-7):
        - Wait for barrier file
        - Read shared data from barrier
        - Skip blockchain initialization
        - Proceed directly to mining
    """

    def __init__(self, cache_root: Path, worker_id: int, total_workers: int = 8):
        """
        Initialize worker barrier.

        Args:
            cache_root: Root directory for cache (e.g., ~/.cache/grail)
            worker_id: This worker's ID (0 = leader)
            total_workers: Total number of workers
        """
        self.cache_root = Path(cache_root)
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.is_leader = worker_id == 0

        # Barrier file paths
        self.barrier_dir = self.cache_root / BARRIER_DIR
        self.barrier_file = self.barrier_dir / BARRIER_FILE
        self.run_id_file = self.barrier_dir / RUN_ID_FILE
        self.block_cache_file = self.barrier_dir / BLOCK_CACHE_FILE

        # ALL workers clean up very stale files on startup to prevent
        # 25-hour-old barrier files from blocking new runs
        self._ensure_barrier_dir()
        self._cleanup_very_stale_files()

        # Generate or read run ID
        if self.is_leader:
            # Leader generates new run ID and cleans up stale files
            self.run_id = str(uuid.uuid4())[:8]  # Short unique ID
            self._cleanup_stale_barrier()
            # Write run ID immediately so followers can read it
            with open(self.run_id_file, "w") as f:
                f.write(self.run_id)
            logger.info("Leader started run_id=%s", self.run_id)
        else:
            # Follower reads run ID (will wait if not yet available)
            self.run_id: str | None = None

        logger.info(
            "WorkerBarrier initialized: worker_id=%d, is_leader=%s, barrier_file=%s",
            worker_id,
            self.is_leader,
            self.barrier_file,
        )

    def _cleanup_stale_barrier(self) -> None:
        """Leader removes any existing barrier file on startup."""
        if self.barrier_file.exists():
            try:
                self.barrier_file.unlink()
                logger.info("Leader removed stale barrier file from previous run")
            except IOError as e:
                logger.warning("Failed to remove stale barrier file: %s", e)

    def _cleanup_very_stale_files(self) -> None:
        """
        Remove very stale files on ANY worker startup.

        This prevents scenarios where a 25-hour-old barrier file blocks new runs.
        Files older than BARRIER_STALE_THRESHOLD (1 hour) are deleted.
        """
        if not self.barrier_dir.exists():
            return

        try:
            now = time.time()
            removed = 0

            for f in self.barrier_dir.iterdir():
                try:
                    # Check file modification time
                    mtime = f.stat().st_mtime
                    age = now - mtime

                    if age > BARRIER_STALE_THRESHOLD:
                        f.unlink()
                        removed += 1
                        logger.info(
                            "Worker %d removed very stale file: %s (%.0f hours old)",
                            self.worker_id,
                            f.name,
                            age / 3600,
                        )
                except (IOError, OSError):
                    pass  # Skip files we can't access

            if removed > 0:
                logger.info(
                    "Worker %d cleaned up %d stale barrier files on startup",
                    self.worker_id,
                    removed,
                )
        except Exception as e:
            logger.warning("Failed to cleanup stale files: %s", e)

    def _ensure_barrier_dir(self) -> None:
        """Create barrier directory if needed."""
        self.barrier_dir.mkdir(parents=True, exist_ok=True)

    def _is_barrier_fresh(self, data: dict[str, Any]) -> bool:
        """Check if barrier data is fresh enough to use."""
        timestamp = data.get("timestamp", 0)
        age = time.time() - timestamp
        if age > BARRIER_MAX_AGE:
            logger.warning(
                "Barrier file is stale (%.0fs old, max %.0fs)",
                age,
                BARRIER_MAX_AGE,
            )
            return False
        return True

    def signal_ready(self, shared_data: dict[str, Any]) -> None:
        """
        Leader signals ready with shared data.

        Args:
            shared_data: Data to share with followers (trainer_bucket, checkpoint_path, etc.)
        """
        if not self.is_leader:
            logger.warning("Only leader (worker 0) should signal ready")
            return

        self._ensure_barrier_dir()

        # Add metadata
        barrier_data = {
            **shared_data,
            "timestamp": time.time(),
            "leader_worker_id": self.worker_id,
            "leader_pid": os.getpid(),
        }

        # Write atomically (write to temp, then rename)
        temp_file = self.barrier_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(barrier_data, f, indent=2)
        temp_file.rename(self.barrier_file)

        logger.info(
            "Leader signaled ready: barrier_file=%s, data_keys=%s",
            self.barrier_file,
            list(shared_data.keys()),
        )

    async def wait_for_leader(self, timeout: float = 300) -> dict[str, Any]:
        """
        Follower waits for leader's ready signal.

        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)

        Returns:
            Shared data from leader

        Raises:
            TimeoutError: If leader doesn't signal within timeout
        """
        if self.is_leader:
            logger.warning("Leader should not wait for itself")
            return {}

        start_time = time.time()
        poll_interval = 1.0  # Start with 1 second
        max_poll_interval = 10.0

        logger.info(
            "Worker %d waiting for leader (timeout=%.0fs)...",
            self.worker_id,
            timeout,
        )

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Worker {self.worker_id} timed out waiting for leader "
                    f"after {elapsed:.0f}s"
                )

            # Check for barrier file
            if self.barrier_file.exists():
                try:
                    with open(self.barrier_file) as f:
                        data = json.load(f)

                    # Verify freshness
                    if self._is_barrier_fresh(data):
                        logger.info(
                            "Worker %d received leader signal (waited %.1fs)",
                            self.worker_id,
                            elapsed,
                        )
                        return data
                    else:
                        # Check if VERY stale (hours old) - delete and wait for fresh one
                        timestamp = data.get("timestamp", 0)
                        age = time.time() - timestamp
                        if age > BARRIER_STALE_THRESHOLD:
                            logger.warning(
                                "Worker %d deleting very stale barrier file (%.1f hours old)",
                                self.worker_id,
                                age / 3600,
                            )
                            try:
                                self.barrier_file.unlink()
                            except IOError:
                                pass
                        else:
                            logger.debug("Barrier file exists but is stale, waiting...")
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug("Error reading barrier file: %s", e)

            # Wait with exponential backoff (capped)
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, max_poll_interval)

            # Log progress periodically
            if int(elapsed) % 30 == 0 and elapsed > 0:
                logger.info(
                    "Worker %d still waiting for leader (%.0fs elapsed)...",
                    self.worker_id,
                    elapsed,
                )

    def signal_checkpoint_downloading(self, checkpoint_window: int) -> None:
        """
        Leader signals that it's downloading a new checkpoint.

        Followers should wait before generating if they see this.
        """
        if not self.is_leader:
            return

        if not self.barrier_file.exists():
            logger.warning("Cannot signal downloading: barrier file doesn't exist")
            return

        try:
            with open(self.barrier_file) as f:
                data = json.load(f)

            data["downloading_checkpoint"] = checkpoint_window
            data["download_started_at"] = time.time()

            # Write atomically
            temp_file = self.barrier_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.barrier_file)

            logger.info(
                "Leader signaling: downloading checkpoint %d",
                checkpoint_window,
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to signal downloading: %s", e)

    def update_checkpoint(self, checkpoint_path: str, checkpoint_window: int) -> None:
        """
        Leader updates the checkpoint info in barrier file.

        Called when a new checkpoint is downloaded. Clears downloading state.
        """
        if not self.is_leader:
            return

        if not self.barrier_file.exists():
            logger.warning("Cannot update checkpoint: barrier file doesn't exist")
            return

        try:
            with open(self.barrier_file) as f:
                data = json.load(f)

            data["checkpoint_path"] = checkpoint_path
            data["checkpoint_window"] = checkpoint_window
            data["checkpoint_updated_at"] = time.time()
            # Clear downloading state
            data.pop("downloading_checkpoint", None)
            data.pop("download_started_at", None)

            # Write atomically
            temp_file = self.barrier_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.barrier_file)

            logger.info(
                "Leader updated checkpoint in barrier: %s (window %d)",
                checkpoint_path,
                checkpoint_window,
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to update checkpoint in barrier: %s", e)

    def signal_current_window(self, window: int) -> None:
        """
        Leader signals which window it's currently working on.

        Followers can use this to avoid jumping ahead of the leader.
        """
        if not self.is_leader:
            return

        if not self.barrier_file.exists():
            return

        try:
            with open(self.barrier_file) as f:
                data = json.load(f)

            data["current_window"] = window
            data["window_updated_at"] = time.time()

            # Write atomically
            temp_file = self.barrier_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.barrier_file)
        except (json.JSONDecodeError, IOError):
            pass  # Non-critical, don't spam logs

    def get_leader_window(self) -> int | None:
        """
        Get the window the leader is currently working on.

        Returns:
            Leader's current window, or None if not available
        """
        if not self.barrier_file.exists():
            return None

        try:
            with open(self.barrier_file) as f:
                data = json.load(f)
            return data.get("current_window")
        except (json.JSONDecodeError, IOError):
            return None

    def update_shared_block(self, block: int) -> None:
        """
        Leader updates the shared block number for followers.

        This is a lightweight operation - just writes block number and timestamp
        to a small text file. Followers read this instead of making RPC calls.

        Args:
            block: Current block number from blockchain
        """
        if not self.is_leader:
            return

        try:
            # Format: "block_number timestamp"
            content = f"{block} {time.time():.3f}"
            # Write atomically
            temp_file = self.block_cache_file.with_suffix(".tmp")
            temp_file.write_text(content)
            temp_file.rename(self.block_cache_file)
        except IOError:
            pass  # Non-critical

    def get_shared_block(self, max_age: float = BLOCK_CACHE_MAX_AGE) -> int | None:
        """
        Get the shared block number from leader's cache.

        Followers use this to avoid redundant blockchain RPC calls.
        Returns None if cache is stale or unavailable.

        Args:
            max_age: Maximum age in seconds before cache is considered stale

        Returns:
            Block number if fresh cache exists, None otherwise
        """
        if not self.block_cache_file.exists():
            return None

        try:
            content = self.block_cache_file.read_text().strip()
            parts = content.split()
            if len(parts) != 2:
                return None

            block = int(parts[0])
            timestamp = float(parts[1])

            # Check freshness
            age = time.time() - timestamp
            if age > max_age:
                return None

            return block
        except (IOError, ValueError):
            return None

    async def wait_for_leader_window(
        self, target_window: int, timeout: float = 120.0
    ) -> bool:
        """
        Follower waits until leader reaches or passes target window.

        Args:
            target_window: Wait until leader's window >= this value
            timeout: Maximum time to wait in seconds

        Returns:
            True if leader caught up, False if timeout
        """
        if self.is_leader:
            return True

        start_time = time.time()
        poll_interval = 1.0

        while True:
            leader_window = self.get_leader_window()
            if leader_window is not None and leader_window >= target_window:
                return True

            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(
                    "Worker %d timeout waiting for leader to reach window %d (leader at %s)",
                    self.worker_id,
                    target_window,
                    leader_window,
                )
                return False

            if int(elapsed) % 10 == 0 and elapsed > 0:
                logger.info(
                    "Worker %d waiting for leader to reach window %d (leader at %s, %.0fs)...",
                    self.worker_id,
                    target_window,
                    leader_window,
                    elapsed,
                )

            await asyncio.sleep(poll_interval)

    async def wait_for_checkpoint_sync(
        self, my_checkpoint_window: int | None, timeout: float = 300.0
    ) -> int | None:
        """
        Follower waits if leader is downloading a newer checkpoint.

        Args:
            my_checkpoint_window: Follower's current checkpoint window
            timeout: Maximum time to wait in seconds

        Returns:
            The checkpoint window to use, or None if should continue with current
        """
        if self.is_leader:
            return my_checkpoint_window

        if not self.barrier_file.exists():
            return my_checkpoint_window

        start_time = time.time()
        poll_interval = 1.0

        while True:
            try:
                with open(self.barrier_file) as f:
                    data = json.load(f)

                leader_checkpoint = data.get("checkpoint_window")
                downloading_checkpoint = data.get("downloading_checkpoint")

                # If leader is downloading a checkpoint newer than ours, wait
                if downloading_checkpoint is not None:
                    if my_checkpoint_window is None or downloading_checkpoint > my_checkpoint_window:
                        elapsed = time.time() - start_time
                        if elapsed > timeout:
                            logger.warning(
                                "Timeout waiting for leader to download checkpoint %d",
                                downloading_checkpoint,
                            )
                            return my_checkpoint_window

                        if int(elapsed) % 10 == 0 and elapsed > 0:
                            logger.info(
                                "Worker %d waiting for leader to download checkpoint %d (%.0fs)...",
                                self.worker_id,
                                downloading_checkpoint,
                                elapsed,
                            )
                        await asyncio.sleep(poll_interval)
                        continue

                # Leader finished downloading or no download in progress
                if leader_checkpoint is not None:
                    if my_checkpoint_window is None or leader_checkpoint > my_checkpoint_window:
                        logger.info(
                            "Worker %d syncing to leader's checkpoint %d (was %s)",
                            self.worker_id,
                            leader_checkpoint,
                            my_checkpoint_window,
                        )
                        return leader_checkpoint

                return my_checkpoint_window

            except (json.JSONDecodeError, IOError):
                return my_checkpoint_window

    def get_checkpoint_info(self) -> tuple[str | None, int | None]:
        """
        Follower gets current checkpoint info from barrier.

        Returns:
            (checkpoint_path, checkpoint_window) or (None, None) if not available
        """
        if not self.barrier_file.exists():
            return None, None

        try:
            with open(self.barrier_file) as f:
                data = json.load(f)
            return data.get("checkpoint_path"), data.get("checkpoint_window")
        except (json.JSONDecodeError, IOError):
            return None, None

    def is_leader_downloading(self, my_checkpoint_window: int | None = None) -> bool:
        """
        Non-blocking check if leader is currently downloading a checkpoint.

        Args:
            my_checkpoint_window: If provided, only returns True if leader is
                                  downloading a NEWER checkpoint than this.

        Returns:
            True if leader is downloading (and it's newer than ours, if specified)
        """
        if self.is_leader:
            return False

        if not self.barrier_file.exists():
            return False

        try:
            with open(self.barrier_file) as f:
                data = json.load(f)

            downloading_checkpoint = data.get("downloading_checkpoint")
            if downloading_checkpoint is None:
                return False

            # If we have a checkpoint window, only abort if leader's is newer
            if my_checkpoint_window is not None:
                return downloading_checkpoint > my_checkpoint_window

            return True
        except (json.JSONDecodeError, IOError):
            return False

    def cleanup(self) -> None:
        """
        Leader cleans up barrier file on shutdown.
        """
        if not self.is_leader:
            return

        try:
            if self.barrier_file.exists():
                self.barrier_file.unlink()
                logger.info("Leader cleaned up barrier file")
        except IOError as e:
            logger.warning("Failed to cleanup barrier file: %s", e)


def get_worker_config() -> tuple[int, int]:
    """
    Get worker configuration from environment.

    Returns:
        (worker_id, total_workers)
    """
    worker_id = int(os.getenv("GRAIL_WORKER_ID", "0"))
    total_workers = int(os.getenv("GRAIL_TOTAL_WORKERS", "1"))
    return worker_id, total_workers


def is_leader_worker() -> bool:
    """Check if this is the leader worker (worker 0)."""
    worker_id, _ = get_worker_config()
    return worker_id == 0


# --------------------------------------------------------------------------- #
#                   Rollout Staging for Multi-Worker Aggregation              #
# --------------------------------------------------------------------------- #

ROLLOUT_STAGING_DIR = ".rollout-staging"
ROLLOUT_STAGING_MAX_AGE = 300  # 5 minutes - older files are stale
ROLLOUT_STAGING_STALE_THRESHOLD = 3600  # 1 hour - delete files older than this on any worker startup


class RolloutStaging:
    """
    Local staging for multi-worker rollout aggregation.

    In multi-worker setups:
    - All workers save rollouts to local staging files
    - Workers 1-7 signal completion and skip R2 upload
    - Worker 0 aggregates all workers' rollouts and uploads once

    This ensures all rollouts end up in a single file for the validator.
    """

    def __init__(self, cache_root: Path, worker_id: int, total_workers: int = 8):
        self.cache_root = Path(cache_root)
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.is_leader = worker_id == 0

        self.staging_dir = self.cache_root / ROLLOUT_STAGING_DIR
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # ALL workers clean up very stale staging files on startup
        # This prevents hours-old files from causing issues
        self._cleanup_very_stale_staging_files()

        # Leader also cleans up ALL staging files on startup to prevent
        # duplicate nonce errors from previous runs
        if self.is_leader:
            self._cleanup_all_staging_files()

    def _cleanup_very_stale_staging_files(self) -> None:
        """
        Remove very stale staging files on ANY worker startup.

        This prevents scenarios where hours-old staging files cause issues.
        Files older than ROLLOUT_STAGING_STALE_THRESHOLD (1 hour) are deleted.
        """
        try:
            now = time.time()
            removed = 0

            for f in self.staging_dir.iterdir():
                if f.suffix not in (".json", ".done", ".tmp", ".uploaded"):
                    continue
                try:
                    mtime = f.stat().st_mtime
                    age = now - mtime

                    if age > ROLLOUT_STAGING_STALE_THRESHOLD:
                        f.unlink()
                        removed += 1
                        logger.debug(
                            "Worker %d removed very stale staging file: %s (%.1f hours old)",
                            self.worker_id,
                            f.name,
                            age / 3600,
                        )
                except (IOError, OSError):
                    pass

            if removed > 0:
                logger.info(
                    "Worker %d cleaned up %d very stale staging files on startup",
                    self.worker_id,
                    removed,
                )
        except Exception as e:
            logger.warning("Failed to cleanup very stale staging files: %s", e)

    def _cleanup_all_staging_files(self) -> None:
        """Leader removes all staging files on startup to prevent stale data mixing."""
        try:
            removed = 0
            for f in self.staging_dir.iterdir():
                if f.suffix in (".json", ".done", ".tmp", ".uploaded"):
                    f.unlink()
                    removed += 1
            if removed > 0:
                logger.info("Leader cleaned up %d stale staging files on startup", removed)
        except Exception as e:
            logger.warning("Failed to cleanup staging files: %s", e)

    def _get_staging_file(self, window: int, wid: int) -> Path:
        """Get path for a worker's staging file."""
        return self.staging_dir / f"window-{window}-worker-{wid}.json"

    def _get_done_file(self, window: int, wid: int) -> Path:
        """Get path for a worker's done marker file."""
        return self.staging_dir / f"window-{window}-worker-{wid}.done"

    def save_rollouts(self, window: int, rollouts: list[dict]) -> None:
        """Save rollouts to local staging file."""
        staging_file = self._get_staging_file(window, self.worker_id)
        done_file = self._get_done_file(window, self.worker_id)

        # Write rollouts atomically
        temp_file = staging_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump({
                "window": window,
                "worker_id": self.worker_id,
                "rollout_count": len(rollouts),
                "rollouts": rollouts,
                "timestamp": time.time(),
            }, f)
        temp_file.rename(staging_file)

        # Write done marker
        with open(done_file, "w") as f:
            f.write(str(time.time()))

        logger.info(
            "Worker %d staged %d rollouts for window %d",
            self.worker_id, len(rollouts), window,
        )

    def load_rollouts(self, window: int, worker_id: int) -> list[dict]:
        """Load rollouts from a worker's staging file."""
        staging_file = self._get_staging_file(window, worker_id)
        if not staging_file.exists():
            return []

        try:
            with open(staging_file) as f:
                data = json.load(f)

            # Check if stale
            age = time.time() - data.get("timestamp", 0)
            if age > ROLLOUT_STAGING_MAX_AGE:
                logger.warning(
                    "Stale staging file for worker %d window %d (%.0fs old)",
                    worker_id, window, age,
                )
                return []

            return data.get("rollouts", [])
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load staging file for worker %d: %s", worker_id, e)
            return []

    def is_worker_done(self, window: int, worker_id: int) -> bool:
        """Check if a worker has completed staging for a window."""
        done_file = self._get_done_file(window, worker_id)
        return done_file.exists()

    async def wait_for_workers(
        self, window: int, timeout: float = 30.0, min_workers: int | None = None,
        early_exit_threshold: float = 0.85,
    ) -> int:
        """
        Leader waits for other workers to finish staging.

        Uses fast polling (0.2s) and early exit when threshold reached.

        Args:
            window: Window number to wait for
            timeout: Max time to wait in seconds
            min_workers: Minimum workers to wait for (default: total_workers - 1)
            early_exit_threshold: Exit early if this fraction done (0.85 = 85%)

        Returns:
            Number of workers that completed staging
        """
        if not self.is_leader:
            return 0

        if min_workers is None:
            min_workers = self.total_workers - 1

        # Calculate early exit count
        early_exit_count = max(1, int((self.total_workers - 1) * early_exit_threshold))

        start_time = time.time()
        poll_interval = 0.2  # Fast polling
        last_count = 0
        stall_start = None
        stall_timeout = 5.0  # Exit if no progress for 5s after threshold

        while True:
            elapsed = time.time() - start_time
            done_count = sum(
                1 for wid in range(1, self.total_workers)
                if self.is_worker_done(window, wid)
            )

            if done_count >= min_workers:
                logger.info(
                    "Leader: %d/%d workers completed staging for window %d (waited %.1fs)",
                    done_count, self.total_workers - 1, window, elapsed,
                )
                return done_count

            # Early exit: if enough workers done and progress stalled
            if done_count >= early_exit_count:
                if done_count > last_count:
                    stall_start = time.time()
                    last_count = done_count
                elif stall_start is not None:
                    stall_elapsed = time.time() - stall_start
                    if stall_elapsed > stall_timeout:
                        logger.info(
                            "Leader: early exit with %d/%d workers (%.0f%%) for window %d "
                            "(no progress for %.1fs)",
                            done_count, self.total_workers - 1,
                            100 * done_count / (self.total_workers - 1),
                            window, stall_elapsed,
                        )
                        return done_count
                else:
                    stall_start = time.time()
                    last_count = done_count

            if elapsed > timeout:
                logger.warning(
                    "Leader: timeout waiting for workers, only %d/%d ready for window %d",
                    done_count, self.total_workers - 1, window,
                )
                return done_count

            await asyncio.sleep(poll_interval)

    def aggregate_rollouts(self, window: int) -> list[dict]:
        """
        Leader aggregates rollouts from all workers.

        Returns:
            Combined list of rollouts from all workers, sorted by problem_index
            to ensure file position matches validator's group_index expectation.
            Duplicates are detected and removed to prevent nonce collisions.
        """
        if not self.is_leader:
            return []

        all_rollouts = []
        for wid in range(self.total_workers):
            rollouts = self.load_rollouts(window, wid)
            if rollouts:
                all_rollouts.extend(rollouts)
                logger.debug(
                    "Aggregated %d rollouts from worker %d for window %d",
                    len(rollouts), wid, window,
                )

        # SAFETY NET: Deduplicate by (rollout_group, rollout_index) to prevent duplicate nonces.
        # This catches any edge cases where the same problem was processed twice (e.g., due to
        # file locking issues in Docker, race conditions, or stale staging files).
        if all_rollouts:
            seen_keys: set[tuple[int, int]] = set()
            unique_rollouts = []
            duplicate_count = 0

            for r in all_rollouts:
                key = (r.get("rollout_group", 0), r.get("rollout_index", 0))
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_rollouts.append(r)
                else:
                    duplicate_count += 1

            if duplicate_count > 0:
                logger.warning(
                    "⚠️ DUPLICATES DETECTED: Removed %d duplicate rollouts (kept %d unique)",
                    duplicate_count,
                    len(unique_rollouts),
                )
            all_rollouts = unique_rollouts

        # CRITICAL: Sort by problem_index (rollout_group) to match validator expectations.
        # Validator uses file position as group_index for seed derivation.
        # Without sorting, rollouts would be in worker order (W0, W1, W2...) instead of
        # problem order (P0, P1, P2...), causing env_prompt_valid check failures.
        all_rollouts.sort(key=lambda r: r.get("rollout_group", 0))

        # Detect gaps and TRUNCATE at first gap to ensure contiguous indices.
        # Validator uses file position as group_index - any gap causes all subsequent
        # rollouts to fail validation because their file position won't match their
        # actual problem_index.
        if all_rollouts:
            problem_indices = sorted(set(r.get("rollout_group", 0) for r in all_rollouts))

            # Find first gap (first missing index starting from 0)
            first_gap = None
            for expected_idx in range(problem_indices[-1] + 1):
                if expected_idx not in problem_indices:
                    first_gap = expected_idx
                    break

            if first_gap is not None:
                # Truncate: keep only rollouts with problem_index < first_gap
                original_count = len(all_rollouts)
                all_rollouts = [r for r in all_rollouts if r.get("rollout_group", 0) < first_gap]
                truncated_count = original_count - len(all_rollouts)

                logger.warning(
                    "⚠️ GAP at problem %d: Truncated %d rollouts (keeping %d contiguous problems [0-%d])",
                    first_gap,
                    truncated_count,
                    first_gap,
                    first_gap - 1,
                )
            else:
                logger.info(
                    "✅ No gaps: %d contiguous problems [%d-%d]",
                    len(problem_indices),
                    problem_indices[0],
                    problem_indices[-1],
                )

        logger.info(
            "Leader aggregated %d total rollouts from %d workers for window %d (sorted by problem_index)",
            len(all_rollouts), self.total_workers, window,
        )
        return all_rollouts

    def _get_uploaded_file(self, window: int) -> Path:
        """Get path for the window uploaded marker file."""
        return self.staging_dir / f"window-{window}.uploaded"

    def mark_window_uploaded(self, window: int) -> None:
        """Leader marks window as uploaded so late workers skip staging."""
        if not self.is_leader:
            return
        uploaded_file = self._get_uploaded_file(window)
        with open(uploaded_file, "w") as f:
            f.write(str(time.time()))
        logger.debug("Marked window %d as uploaded", window)

    def is_window_uploaded(self, window: int) -> bool:
        """Check if a window has already been uploaded."""
        return self._get_uploaded_file(window).exists()

    def cleanup_window(self, window: int) -> None:
        """Clean up staging files for a completed window."""
        for wid in range(self.total_workers):
            for file in [
                self._get_staging_file(window, wid),
                self._get_done_file(window, wid),
            ]:
                try:
                    if file.exists():
                        file.unlink()
                except IOError:
                    pass
        # Also clean up uploaded marker
        try:
            uploaded_file = self._get_uploaded_file(window)
            if uploaded_file.exists():
                uploaded_file.unlink()
        except IOError:
            pass


# --------------------------------------------------------------------------- #
#                   Redis-Based Multi-Node Rollout Aggregation                #
# --------------------------------------------------------------------------- #

REDIS_ROLLOUT_PREFIX = "grail:rollouts"
REDIS_ROLLOUT_TTL = 600  # 10 minutes - enough for window + upload
# Node ID key PREFIX - actual key is per-hostname to prevent cross-node collision
REDIS_NODE_ID_KEY_PREFIX = "grail:config:node_id"
REDIS_NODE_ID_TTL = 86400  # 24 hours - persist across restarts


def _get_node_id_key() -> str:
    """Get hostname-specific Redis key for node_id sharing.

    Each physical node uses a different key (grail:config:node_id:{hostname})
    so workers on different nodes don't overwrite each other's node_id.
    """
    import socket
    hostname = socket.gethostname()
    return f"{REDIS_NODE_ID_KEY_PREFIX}:{hostname}"


REDIS_CHECKPOINT_KEY = "grail:checkpoint:current"  # Current checkpoint info
REDIS_CHECKPOINT_TTL = 600  # 10 minutes


class RedisRolloutAggregator:
    """
    Redis-based aggregation for multi-node/multi-worker deployments.

    SINGLE-NODE MODE (8 GPUs, all workers push directly to Redis):
        Worker 0-7 → Each pushes own rollouts to Redis → Worker 0 (hub) aggregates → upload
        - No file-based staging needed
        - Each worker identified by worker_id

    MULTI-NODE MODE (4 nodes × 8 GPUs = 32 GPUs on same wallet):
        Node 1: Workers 0-7 → Each pushes to Redis with node:worker key
        Node 2: Workers 0-7 → Each pushes to Redis with node:worker key
        Node 3: Workers 0-7 → Each pushes to Redis with node:worker key
        Node 4: Workers 0-7 → Each pushes to Redis with node:worker key
                                   ↓
        Hub (Node 1, Worker 0): Aggregates from Redis → Single parquet upload

    Environment variables:
        GRAIL_REDIS_URL - Redis connection URL (required)
        GRAIL_HUB_MODE=1 - Enables hub aggregation (only worker 0 on one node)
        GRAIL_NODE_ID - Unique node identifier (auto-generated if not set)
        GRAIL_TOTAL_NODES - Number of nodes (default: 1 for single-node)
        GRAIL_TOTAL_WORKERS - Workers per node (default: 8)

    Usage:
        # Each worker pushes its own rollouts:
        aggregator = RedisRolloutAggregator(redis_url, node_id, worker_id, is_hub=False)
        aggregator.push_rollouts(window, rollouts)

        # Hub (worker 0) aggregates from all:
        aggregator = RedisRolloutAggregator(redis_url, node_id, worker_id=0, is_hub=True)
        all_rollouts = aggregator.aggregate_from_all(window)
        await upload_window(wallet, window, all_rollouts)
    """

    def __init__(
        self,
        redis_url: str,
        node_id: str | None = None,
        is_hub: bool = False,
        total_nodes: int = 1,
        worker_id: int = 0,
        total_workers: int = 8,
    ):
        """
        Initialize Redis rollout aggregator.

        Args:
            redis_url: Redis connection URL
            node_id: Unique identifier for this node (if None, shared via Redis)
            is_hub: If True, this worker collects and uploads
            total_nodes: Expected number of nodes (1 for single-node)
            worker_id: This worker's ID (0-7)
            total_workers: Workers per node (typically 8)
        """
        import redis

        self.redis_url = redis_url
        self.is_hub = is_hub
        self.total_nodes = total_nodes
        self.worker_id = worker_id
        self.total_workers = total_workers

        # Connect to Redis first (needed for node_id sharing)
        self.client = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis for rollout aggregation: %s", e)
            raise

        # Get or share node_id via Redis
        # Worker 0 sets the shared node_id, other workers read it
        self.node_id = self._get_or_share_node_id(node_id)

        # Unique key for this worker (node:worker)
        self.worker_key = f"{self.node_id}:w{worker_id}"

        logger.info(
            "Redis rollout aggregator connected: node=%s, worker=%d, hub=%s, nodes=%d, workers=%d",
            self.node_id,
            worker_id,
            is_hub,
            total_nodes,
            total_workers,
        )

    def _generate_node_id(self) -> str:
        """Generate a unique node identifier."""
        import socket
        import uuid

        hostname = socket.gethostname()
        short_uuid = str(uuid.uuid4())[:8]
        return f"{hostname}-{short_uuid}"

    def _get_or_share_node_id(self, provided_node_id: str | None) -> str:
        """
        Get shared node_id from Redis or set it if worker 0.

        This ensures all workers on the same node use the same node_id,
        giving cleaner worker keys like node-1:w0, node-1:w1 instead of
        each worker generating its own random UUID.

        Logic:
        - If provided_node_id is set (from GRAIL_NODE_ID env var):
          - Worker 0: Store it in Redis for other workers
          - Other workers: Use the provided value
        - If no provided_node_id:
          - Worker 0: Generate one and store in Redis
          - Other workers: Try to read from Redis, fall back to generate

        Args:
            provided_node_id: Node ID from environment variable (if any)

        Returns:
            The node_id to use (shared across all workers)
        """
        is_leader = self.worker_id == 0

        # Get hostname-specific key for node_id sharing (prevents cross-node collision)
        node_id_key = _get_node_id_key()

        # If explicitly provided via env var, use it
        if provided_node_id:
            if is_leader:
                # Leader stores the provided node_id in Redis for other workers on same host
                try:
                    self.client.setex(node_id_key, REDIS_NODE_ID_TTL, provided_node_id)
                    logger.info(
                        "Worker 0: stored node_id '%s' in Redis key '%s' for other workers",
                        provided_node_id, node_id_key,
                    )
                except Exception as e:
                    logger.warning("Failed to store node_id in Redis: %s", e)
            return provided_node_id

        # No provided node_id - use Redis for coordination
        if is_leader:
            # Leader generates and stores a node_id
            generated_id = self._generate_node_id()
            try:
                # Use SETNX to avoid overwriting if another leader already set it
                # (e.g., if leader restarted but followers are still running)
                set_result = self.client.setnx(node_id_key, generated_id)
                if set_result:
                    self.client.expire(node_id_key, REDIS_NODE_ID_TTL)
                    logger.info(
                        "Worker 0: generated and stored node_id '%s' in Redis key '%s'",
                        generated_id, node_id_key,
                    )
                    return generated_id
                else:
                    # Another process already set it, read and use that one
                    existing_id = self.client.get(node_id_key)
                    if existing_id:
                        logger.info(
                            "Worker 0: using existing node_id '%s' from Redis",
                            existing_id,
                        )
                        return existing_id
                    return generated_id
            except Exception as e:
                logger.warning("Failed to coordinate node_id via Redis: %s", e)
                return generated_id
        else:
            # Follower tries to read from Redis
            try:
                # Wait a bit for leader to set it (up to 5 seconds)
                for _ in range(50):  # 50 * 0.1s = 5 seconds
                    shared_id = self.client.get(node_id_key)
                    if shared_id:
                        logger.info(
                            "Worker %d: using shared node_id '%s' from Redis",
                            self.worker_id,
                            shared_id,
                        )
                        return shared_id
                    time.sleep(0.1)

                # Timeout - generate our own (fallback)
                logger.warning(
                    "Worker %d: no shared node_id in Redis after 5s, generating own",
                    self.worker_id,
                )
                return self._generate_node_id()
            except Exception as e:
                logger.warning("Failed to read node_id from Redis: %s", e)
                return self._generate_node_id()

    def _rollouts_key(self, window: int, worker_key: str) -> str:
        """Get Redis key for a worker's rollouts."""
        return f"{REDIS_ROLLOUT_PREFIX}:{window}:worker:{worker_key}"

    def _ready_key(self, window: int, worker_key: str) -> str:
        """Get Redis key for worker ready marker."""
        return f"{REDIS_ROLLOUT_PREFIX}:{window}:ready:{worker_key}"

    def _workers_list_key(self, window: int) -> str:
        """Get Redis key for list of workers that pushed."""
        return f"{REDIS_ROLLOUT_PREFIX}:{window}:workers"

    def push_rollouts(self, window: int, rollouts: list[dict]) -> bool:
        """
        Push this worker's rollouts to Redis.

        Called by EACH worker (not just leader) to push their own rollouts.

        Args:
            window: Window number
            rollouts: This worker's rollouts

        Returns:
            True if successfully pushed
        """
        try:
            # Serialize rollouts
            data = json.dumps({
                "node_id": self.node_id,
                "worker_id": self.worker_id,
                "worker_key": self.worker_key,
                "window": window,
                "rollout_count": len(rollouts),
                "rollouts": rollouts,
                "timestamp": time.time(),
            })

            # Store with TTL using worker-specific key
            rollouts_key = self._rollouts_key(window, self.worker_key)
            self.client.setex(rollouts_key, REDIS_ROLLOUT_TTL, data)

            # Register this worker
            workers_key = self._workers_list_key(window)
            self.client.sadd(workers_key, self.worker_key)
            self.client.expire(workers_key, REDIS_ROLLOUT_TTL)

            # Set ready marker
            ready_key = self._ready_key(window, self.worker_key)
            self.client.setex(ready_key, REDIS_ROLLOUT_TTL, str(time.time()))

            logger.info(
                "Worker %s pushed %d rollouts to Redis for window %d",
                self.worker_key, len(rollouts), window,
            )
            return True

        except Exception as e:
            logger.error("Failed to push rollouts to Redis: %s", e)
            return False

    async def wait_for_workers(
        self, window: int, timeout: float = 60.0, min_workers: int | None = None,
        early_exit_threshold: float = 0.90,
    ) -> int:
        """
        Hub waits for workers to push their rollouts.

        Uses fast polling (0.3s) and early exit when threshold reached.
        Also uses Redis pub/sub for instant notification when available.

        Args:
            window: Window number
            timeout: Max time to wait
            min_workers: Minimum workers required (default: total_nodes * total_workers)
            early_exit_threshold: Exit early if this fraction of workers ready (0.90 = 90%)

        Returns:
            Number of workers that pushed rollouts
        """
        if not self.is_hub:
            return 0

        expected_workers = self.total_nodes * self.total_workers
        if min_workers is None:
            min_workers = expected_workers

        # Calculate early exit count (e.g., 90% of expected)
        early_exit_count = max(1, int(expected_workers * early_exit_threshold))

        start_time = time.time()
        poll_interval = 0.3  # Fast polling for quicker detection
        last_count = 0
        stall_start = None
        stall_timeout = 10.0  # Exit if no progress for 10s after early threshold

        while True:
            elapsed = time.time() - start_time

            # Count ready workers
            workers_key = self._workers_list_key(window)
            ready_workers = self.client.scard(workers_key)

            # Full coverage - all workers ready
            if ready_workers >= min_workers:
                logger.info(
                    "Hub: %d/%d workers ready for window %d (waited %.1fs)",
                    ready_workers, expected_workers, window, elapsed,
                )
                return ready_workers

            # Early exit: if we have enough workers and progress stalled
            if ready_workers >= early_exit_count:
                if ready_workers > last_count:
                    # Progress - reset stall timer
                    stall_start = time.time()
                    last_count = ready_workers
                elif stall_start is not None:
                    stall_elapsed = time.time() - stall_start
                    if stall_elapsed > stall_timeout:
                        logger.info(
                            "Hub: early exit with %d/%d workers (%.0f%%) for window %d "
                            "(no progress for %.1fs, total wait %.1fs)",
                            ready_workers, expected_workers,
                            100 * ready_workers / expected_workers,
                            window, stall_elapsed, elapsed,
                        )
                        return ready_workers
                else:
                    stall_start = time.time()
                    last_count = ready_workers

            if elapsed > timeout:
                logger.warning(
                    "Hub: timeout waiting for workers, only %d/%d ready for window %d",
                    ready_workers, expected_workers, window,
                )
                return ready_workers

            await asyncio.sleep(poll_interval)

    # Backward compatibility alias
    async def wait_for_nodes(
        self, window: int, timeout: float = 60.0, min_nodes: int | None = None
    ) -> int:
        """Deprecated: use wait_for_workers instead."""
        min_workers = min_nodes * self.total_workers if min_nodes else None
        return await self.wait_for_workers(window, timeout, min_workers)

    def aggregate_from_workers(self, window: int) -> list[dict]:
        """
        Hub aggregates rollouts from all workers.

        Returns:
            Combined list of rollouts from all workers, sorted by problem_index
        """
        if not self.is_hub:
            logger.error("aggregate_from_workers called on non-hub worker")
            return []

        try:
            # Get list of workers that pushed
            workers_key = self._workers_list_key(window)
            worker_keys = self.client.smembers(workers_key)

            if not worker_keys:
                logger.warning("No workers have pushed rollouts for window %d", window)
                return []

            all_rollouts = []
            for worker_key in worker_keys:
                rollouts_key = self._rollouts_key(window, worker_key)
                data = self.client.get(rollouts_key)

                if data:
                    try:
                        parsed = json.loads(data)
                        worker_rollouts = parsed.get("rollouts", [])
                        all_rollouts.extend(worker_rollouts)
                        logger.debug(
                            "Hub aggregated %d rollouts from worker %s",
                            len(worker_rollouts), worker_key,
                        )
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse rollouts from worker %s: %s", worker_key, e)

            # Deduplicate by (rollout_group, rollout_index)
            if all_rollouts:
                seen_keys: set[tuple[int, int]] = set()
                unique_rollouts = []
                duplicate_count = 0

                for r in all_rollouts:
                    key = (r.get("rollout_group", 0), r.get("rollout_index", 0))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unique_rollouts.append(r)
                    else:
                        duplicate_count += 1

                if duplicate_count > 0:
                    logger.warning(
                        "Hub: removed %d duplicate rollouts from aggregation",
                        duplicate_count,
                    )
                all_rollouts = unique_rollouts

            # Sort by problem_index for validator
            all_rollouts.sort(key=lambda r: r.get("rollout_group", 0))

            # Truncate at first gap - validator requires contiguous problem indices
            if all_rollouts:
                problem_indices = sorted(set(r.get("rollout_group", 0) for r in all_rollouts))
                first_gap = None
                for expected_idx in range(problem_indices[-1] + 1):
                    if expected_idx not in problem_indices:
                        first_gap = expected_idx
                        break

                if first_gap is not None:
                    original_count = len(all_rollouts)
                    all_rollouts = [r for r in all_rollouts if r.get("rollout_group", 0) < first_gap]
                    logger.warning(
                        "Hub: gap at problem %d, truncated from %d to %d contiguous rollouts (lost %d)",
                        first_gap, original_count, len(all_rollouts), original_count - len(all_rollouts),
                    )

            logger.info(
                "Hub aggregated %d total rollouts from %d workers for window %d",
                len(all_rollouts), len(worker_keys), window,
            )
            return all_rollouts

        except Exception as e:
            logger.error("Hub failed to aggregate rollouts: %s", e)
            return []

    # Backward compatibility alias
    def aggregate_from_nodes(self, window: int) -> list[dict]:
        """Deprecated: use aggregate_from_workers instead."""
        return self.aggregate_from_workers(window)

    def cleanup_window(self, window: int) -> None:
        """Clean up Redis keys for a completed window."""
        try:
            # Get all workers that pushed
            workers_key = self._workers_list_key(window)
            worker_keys = self.client.smembers(workers_key)

            # Delete all keys for this window
            keys_to_delete = [workers_key]
            for worker_key in worker_keys:
                keys_to_delete.append(self._rollouts_key(window, worker_key))
                keys_to_delete.append(self._ready_key(window, worker_key))

            if keys_to_delete:
                self.client.delete(*keys_to_delete)
                logger.debug("Hub cleaned up %d Redis keys for window %d", len(keys_to_delete), window)

        except Exception as e:
            logger.warning("Failed to cleanup Redis keys for window %d: %s", window, e)

    # --------------------------------------------------------------------------- #
    #                   Cross-Node Checkpoint Synchronization                      #
    # --------------------------------------------------------------------------- #

    def broadcast_checkpoint(self, checkpoint_window: int, checkpoint_path: str) -> None:
        """
        Hub broadcasts checkpoint info to Redis for instant cross-node sync.

        Called by hub when a new checkpoint is ready. Other nodes' worker0s
        can read this instead of waiting for file barrier updates.

        Args:
            checkpoint_window: Window number of the checkpoint
            checkpoint_path: Path to the checkpoint
        """
        try:
            data = json.dumps({
                "checkpoint_window": checkpoint_window,
                "checkpoint_path": checkpoint_path,
                "node_id": self.node_id,
                "timestamp": time.time(),
            })
            self.client.setex(REDIS_CHECKPOINT_KEY, REDIS_CHECKPOINT_TTL, data)
            logger.info(
                "📡 Broadcast checkpoint %d to Redis (path=%s)",
                checkpoint_window, checkpoint_path,
            )
        except Exception as e:
            logger.warning("Failed to broadcast checkpoint to Redis: %s", e)

    def get_redis_checkpoint(self) -> tuple[int | None, str | None]:
        """
        Get current checkpoint info from Redis.

        Returns:
            (checkpoint_window, checkpoint_path) or (None, None) if not available
        """
        try:
            data = self.client.get(REDIS_CHECKPOINT_KEY)
            if data:
                parsed = json.loads(data)
                return parsed.get("checkpoint_window"), parsed.get("checkpoint_path")
            return None, None
        except Exception:
            return None, None

    async def wait_for_checkpoint_update(
        self, current_window: int | None, timeout: float = 30.0
    ) -> tuple[int | None, str | None]:
        """
        Wait for a newer checkpoint to appear in Redis.

        Uses fast polling (0.2s) for quick detection.

        Args:
            current_window: Current checkpoint window (wait for higher)
            timeout: Max time to wait in seconds

        Returns:
            (checkpoint_window, checkpoint_path) or (None, None) if timeout
        """
        start_time = time.time()
        poll_interval = 0.2

        while True:
            checkpoint_window, checkpoint_path = self.get_redis_checkpoint()

            # Found newer checkpoint
            if checkpoint_window is not None:
                if current_window is None or checkpoint_window > current_window:
                    return checkpoint_window, checkpoint_path

            elapsed = time.time() - start_time
            if elapsed > timeout:
                return None, None

            await asyncio.sleep(poll_interval)

    def signal_window_start(self, window: int) -> None:
        """
        Signal that this node is starting work on a window.

        Other nodes can use this to detect if they're behind.
        """
        try:
            key = f"grail:window:active:{self.node_id}"
            self.client.setex(key, REDIS_ROLLOUT_TTL, str(window))
        except Exception:
            pass  # Non-critical

    def get_hub_window(self) -> int | None:
        """
        Get the window that hub (node-1) is currently working on.

        Returns:
            Window number or None if not available
        """
        try:
            # Hub is typically node-1 or the first node
            key = "grail:window:active:node-1"
            value = self.client.get(key)
            return int(value) if value else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Blockchain state caching (reduces RPC load from 16x to 1x)
    # -------------------------------------------------------------------------
    # Leader (worker-0) queries blockchain and caches to Redis
    # Followers read from Redis instead of making independent RPC calls

    BLOCK_CACHE_KEY = "grail:blockchain:current_block"
    BLOCK_CACHE_TTL = 60  # 60 seconds - long enough for any generation cycle

    def cache_current_block(self, block: int) -> None:
        """
        Cache the current block number in Redis.

        Called by leader (worker-0) after querying the blockchain.
        Other workers read this instead of making independent RPC calls.

        Args:
            block: Current block number from blockchain RPC
        """
        try:
            self.client.setex(self.BLOCK_CACHE_KEY, self.BLOCK_CACHE_TTL, str(block))
        except Exception as e:
            logger.debug("Failed to cache current block: %s", e)

    def get_cached_block(self) -> int | None:
        """
        Get the cached current block from Redis.

        Called by followers to avoid independent blockchain RPC calls.

        Returns:
            Cached block number or None if not available/expired
        """
        try:
            value = self.client.get(self.BLOCK_CACHE_KEY)
            return int(value) if value else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Hub ready signal (for cross-node startup coordination)
    # -------------------------------------------------------------------------
    HUB_READY_KEY = "grail:hub:ready"
    HUB_READY_TTL = 300  # 5 minutes - should be renewed during mining

    def signal_hub_ready(self) -> None:
        """
        Signal that the hub is fully initialized and ready.

        Called by hub after completing initialization. Other nodes can check
        this to know when the hub is available for coordination.
        """
        if not self.is_hub:
            return
        try:
            import time
            self.client.setex(self.HUB_READY_KEY, self.HUB_READY_TTL, str(int(time.time())))
            logger.info("🌐 Hub signaled ready to Redis")
        except Exception as e:
            logger.warning("Failed to signal hub ready: %s", e)

    def is_hub_ready(self) -> bool:
        """
        Check if the hub has signaled ready.

        Returns:
            True if hub is ready, False otherwise
        """
        try:
            value = self.client.get(self.HUB_READY_KEY)
            return value is not None
        except Exception:
            return False

    def get_hub_ready_time(self) -> int | None:
        """
        Get the timestamp when hub became ready.

        Returns:
            Unix timestamp or None if not ready
        """
        try:
            value = self.client.get(self.HUB_READY_KEY)
            return int(value) if value else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Generation stop signal (hub tells all workers to stop)
    # -------------------------------------------------------------------------
    # When hub reaches safety blocks, it signals stop. All workers check this
    # and stop generating immediately, ensuring synchronized window completion.

    GENERATION_STOP_KEY = "grail:generation:stop"
    GENERATION_STOP_TTL = 120  # 2 minutes - covers window transition

    def signal_generation_stop(self, window: int) -> None:
        """
        Signal all workers to stop generating for this window.

        Called by hub when it reaches the safety blocks threshold. All other
        workers should check should_stop_generation() and stop immediately.

        Args:
            window: The window to stop generating for
        """
        if not self.is_hub:
            return
        try:
            self.client.setex(self.GENERATION_STOP_KEY, self.GENERATION_STOP_TTL, str(window))
            logger.info("🛑 Hub signaled STOP for window %d → Redis", window)
        except Exception as e:
            logger.warning("Failed to signal generation stop: %s", e)

    def should_stop_generation(self, window: int) -> bool:
        """
        Check if hub has signaled to stop generating for this window.

        Called by non-hub workers in the generation loop. When True, workers
        should immediately stop generating and proceed to upload.

        Args:
            window: The current window being generated

        Returns:
            True if hub has signaled stop for this window, False otherwise
        """
        if self.is_hub:
            return False  # Hub decides for itself, doesn't check this
        try:
            value = self.client.get(self.GENERATION_STOP_KEY)
            if value is None:
                return False
            stop_window = int(value)
            return stop_window == window
        except Exception:
            return False

    def clear_generation_stop(self) -> None:
        """
        Clear the generation stop signal (called at start of new window).

        Hub calls this at the beginning of each window to reset state.
        """
        if not self.is_hub:
            return
        try:
            self.client.delete(self.GENERATION_STOP_KEY)
        except Exception:
            pass  # Non-critical


def get_redis_rollout_aggregator(
    worker_id: int | None = None,
    total_workers: int | None = None,
) -> RedisRolloutAggregator | None:
    """
    Get Redis rollout aggregator if Redis mode is enabled.

    Each worker should call this to get their own aggregator instance.
    Worker 0 with GRAIL_HUB_MODE=1 becomes the hub that aggregates and uploads.

    Environment variables:
        GRAIL_REDIS_URL - Required for Redis mode
        GRAIL_HUB_MODE=1 - This worker is the hub (aggregator/uploader)
        GRAIL_NODE_ID - Optional unique node ID
        GRAIL_TOTAL_NODES - Number of nodes (default: 1 for single-node)
        GRAIL_WORKER_ID - This worker's ID (0-7)
        GRAIL_TOTAL_WORKERS - Workers per node (default: 8)

    Args:
        worker_id: Override for GRAIL_WORKER_ID
        total_workers: Override for GRAIL_TOTAL_WORKERS

    Returns:
        RedisRolloutAggregator if enabled, None otherwise
    """
    redis_url = os.getenv("GRAIL_REDIS_URL", "").strip()
    if not redis_url:
        return None

    try:
        import redis  # noqa: F401

        node_id = os.getenv("GRAIL_NODE_ID")
        total_nodes = int(os.getenv("GRAIL_TOTAL_NODES", "1"))

        # Get worker config
        if worker_id is None:
            worker_id = int(os.getenv("GRAIL_WORKER_ID", "0"))
        if total_workers is None:
            total_workers = int(os.getenv("GRAIL_TOTAL_WORKERS", "8"))

        # Only worker 0 with HUB_MODE=1 is the hub
        is_hub_mode = os.getenv("GRAIL_HUB_MODE", "").lower() in ("1", "true", "yes")
        is_hub = is_hub_mode and worker_id == 0

        return RedisRolloutAggregator(
            redis_url=redis_url,
            node_id=node_id,
            is_hub=is_hub,
            total_nodes=total_nodes,
            worker_id=worker_id,
            total_workers=total_workers,
        )

    except ImportError:
        logger.warning("GRAIL_REDIS_URL set but redis package not installed")
        return None
    except Exception as e:
        logger.warning("Failed to create Redis rollout aggregator: %s", e)
        return None


# --------------------------------------------------------------------------- #
#                   Problem Queue for Gap-Free Distribution                   #
# --------------------------------------------------------------------------- #

PROBLEM_QUEUE_DIR = ".problem-queue"


class ProblemQueue:
    """
    Shared problem queue for gap-free multi-worker problem distribution.

    Instead of round-robin distribution (which creates interleaved gaps when
    workers finish at different times), workers atomically claim the next
    problem index from a shared counter. This guarantees contiguous indices.

    Benefits:
    - No gaps in problem indices (validator requires contiguous indices)
    - Faster workers naturally do more problems (work stealing)
    - Overhead is negligible (~1-5ms per claim vs ~1-2s generation time)

    NOTE: Uses atomic file creation (O_CREAT|O_EXCL) instead of fcntl.flock()
    because flock doesn't work reliably in Docker containers.

    Usage:
        queue = ProblemQueue(cache_root, worker_id)
        while mining:
            problem_index = queue.claim_next_problem(window_start)
            if problem_index < 0:
                break  # Window ended or error
            # Generate rollouts for problem_index
    """

    def __init__(self, cache_root: Path, worker_id: int, node_offset: int = 0):
        """
        Initialize problem queue.

        Args:
            cache_root: Root directory for cache (e.g., ~/.cache/grail)
            worker_id: This worker's ID (0 = leader)
            node_offset: Offset to add to problem indices for multi-node uniqueness
        """
        self.cache_root = Path(cache_root)
        self.worker_id = worker_id
        self.is_leader = worker_id == 0
        self.node_offset = node_offset

        self.queue_dir = self.cache_root / PROBLEM_QUEUE_DIR
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        if node_offset > 0:
            logger.info(
                "ProblemQueue initialized with node_offset=%d (problem indices will start at %d)",
                node_offset,
                node_offset,
            )

        # Leader cleans up stale counter files on startup
        if self.is_leader:
            self._cleanup_stale_counters()

    def _cleanup_stale_counters(self) -> None:
        """Leader removes old counter/claim files on startup."""
        try:
            removed = 0
            for f in self.queue_dir.iterdir():
                if f.suffix in (".counter", ".lock", ".claim"):
                    # Remove files older than 10 minutes
                    try:
                        age = time.time() - f.stat().st_mtime
                        if age > 600:
                            f.unlink()
                            removed += 1
                    except (IOError, OSError):
                        pass
            if removed > 0:
                logger.info("Leader cleaned up %d stale problem queue files", removed)
        except Exception as e:
            logger.warning("Failed to cleanup problem queue files: %s", e)

    def _get_counter_file(self, window: int) -> Path:
        """Get path for window's problem counter file."""
        return self.queue_dir / f"window-{window}.counter"

    def _get_claim_file(self, window: int, problem_index: int) -> Path:
        """Get path for a specific problem's claim file."""
        return self.queue_dir / f"window-{window}-problem-{problem_index}.claim"

    def _get_ready_file(self, window: int) -> Path:
        """Get path for window's ready marker (created by leader after reset)."""
        return self.queue_dir / f"window-{window}.ready"

    def wait_for_leader_reset(self, window: int, timeout: float = 10.0) -> bool:
        """
        Follower waits for leader to finish resetting claim files.

        This prevents a race condition where:
        1. Follower claims problem 0
        2. Leader's reset_counter deletes the claim
        3. Leader claims problem 0
        4. Duplicates!

        Args:
            window: Window start block number
            timeout: Maximum time to wait in seconds

        Returns:
            True if leader reset detected, False if timeout
        """
        if self.is_leader:
            return True  # Leader doesn't wait for itself

        ready_file = self._get_ready_file(window)
        start_time = time.time()
        poll_interval = 0.05  # 50ms

        while time.time() - start_time < timeout:
            if ready_file.exists():
                logger.debug(
                    "Worker %d: leader reset complete for window %d",
                    self.worker_id,
                    window,
                )
                return True
            time.sleep(poll_interval)

        # Timeout - proceed anyway (leader might have crashed)
        logger.warning(
            "Worker %d: timeout waiting for leader reset (window %d), proceeding",
            self.worker_id,
            window,
        )
        return False

    def claim_next_problem(self, window: int, max_attempts: int = MAX_PROBLEMS_PER_WINDOW) -> int:
        """
        Atomically claim the next problem index for a window.

        Uses atomic file creation (O_CREAT|O_EXCL) to ensure only one worker
        claims each index. This works reliably in Docker containers unlike
        fcntl.flock().

        Args:
            window: Window start block number
            max_attempts: Maximum problem index to try (default from GRAIL_MAX_PROBLEMS_PER_WINDOW)

        Returns:
            The claimed problem index (0, 1, 2, ...), or -1 on error
        """
        # Try to claim indices starting from 0
        # Each worker tries to create claim files atomically
        for problem_index in range(max_attempts):
            claim_file = self._get_claim_file(window, problem_index)

            try:
                # Atomic file creation - fails if file already exists
                # O_CREAT | O_EXCL ensures only one process succeeds
                fd = os.open(
                    str(claim_file),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                # Write our worker ID to the claim file
                os.write(fd, f"{self.worker_id}:{time.time()}".encode())
                os.close(fd)

                # Apply node offset for multi-node uniqueness
                global_problem_index = self.node_offset + problem_index
                logger.debug(
                    "Worker %d claimed problem %d (local %d + offset %d) for window %d",
                    self.worker_id,
                    global_problem_index,
                    problem_index,
                    self.node_offset,
                    window,
                )
                return global_problem_index

            except FileExistsError:
                # Another worker already claimed this index, try next
                continue
            except Exception as e:
                logger.error(
                    "Failed to claim problem %d for window %d: %s",
                    problem_index,
                    window,
                    e,
                )
                return -1

        logger.warning(
            "Worker %d: no claimable problems in window %d (tried %d indices)",
            self.worker_id,
            window,
            max_attempts,
        )
        return -1

    def reset_counter(self, window: int) -> None:
        """
        Reset claim files for a new window and signal ready to followers.

        Leader should call this at the start of each window to clean up
        any stale claim files from previous runs. After reset, creates a
        ready marker so followers know it's safe to start claiming.

        IMPORTANT: This must complete BEFORE any follower starts claiming,
        otherwise the leader's reset could delete a follower's claim, causing
        duplicate problem assignments.
        """
        if not self.is_leader:
            return

        try:
            # Remove all claim files for this window
            removed = 0
            pattern = f"window-{window}-problem-"
            for f in self.queue_dir.iterdir():
                if f.name.startswith(pattern) and f.suffix == ".claim":
                    f.unlink()
                    removed += 1

            # Also remove old ready files (from previous windows)
            for f in self.queue_dir.iterdir():
                if f.suffix == ".ready":
                    try:
                        # Keep only recent ready files (last 2 windows)
                        age = time.time() - f.stat().st_mtime
                        if age > 300:  # 5 minutes
                            f.unlink()
                    except (IOError, OSError):
                        pass

            if removed > 0:
                logger.debug("Leader reset %d claim files for window %d", removed, window)

            # Create ready marker so followers can start claiming
            ready_file = self._get_ready_file(window)
            ready_file.write_text(f"{time.time()}")
            logger.debug("Leader signaled ready for window %d", window)

        except IOError as e:
            logger.warning("Failed to reset claims for window %d: %s", window, e)

    def get_current_count(self, window: int) -> int:
        """
        Get current problem count for a window (count claim files).

        Useful for monitoring/logging.
        """
        try:
            pattern = f"window-{window}-problem-"
            count = sum(
                1
                for f in self.queue_dir.iterdir()
                if f.name.startswith(pattern) and f.suffix == ".claim"
            )
            return count
        except (IOError, ValueError):
            return 0

    def cleanup_window(self, window: int) -> None:
        """Clean up claim and ready files for a completed window."""
        try:
            pattern = f"window-{window}-problem-"
            for f in self.queue_dir.iterdir():
                if f.name.startswith(pattern) and f.suffix == ".claim":
                    f.unlink()
            # Also remove the ready file
            ready_file = self._get_ready_file(window)
            if ready_file.exists():
                ready_file.unlink()
        except IOError:
            pass
