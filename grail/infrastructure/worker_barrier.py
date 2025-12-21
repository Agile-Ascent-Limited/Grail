"""
Worker barrier for leader-follower synchronization in multi-GPU mining.

Worker 0 (leader) does initialization (blockchain, checkpoint download) and signals ready.
Workers 1-7 (followers) wait for leader, then proceed with mining only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Barrier file location (relative to cache root)
BARRIER_DIR = ".worker-barrier"
BARRIER_FILE = "leader-ready.json"

# How long barrier data is considered fresh (seconds)
BARRIER_MAX_AGE = 600  # 10 minutes


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

        # Barrier file path
        self.barrier_dir = self.cache_root / BARRIER_DIR
        self.barrier_file = self.barrier_dir / BARRIER_FILE

        logger.info(
            "WorkerBarrier initialized: worker_id=%d, is_leader=%s, barrier_file=%s",
            worker_id,
            self.is_leader,
            self.barrier_file,
        )

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

    def update_checkpoint(self, checkpoint_path: str, checkpoint_window: int) -> None:
        """
        Leader updates the checkpoint info in barrier file.

        Called when a new checkpoint is downloaded.
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

            # Write atomically
            temp_file = self.barrier_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.barrier_file)

            logger.debug(
                "Leader updated checkpoint in barrier: %s (window %d)",
                checkpoint_path,
                checkpoint_window,
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to update checkpoint in barrier: %s", e)

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
