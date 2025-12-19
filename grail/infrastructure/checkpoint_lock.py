"""Cross-process checkpoint locking for multi-worker mining.

When running multiple miner workers on the same machine, only one worker
should download each checkpoint. Other workers wait for the download to
complete and then use the shared cached checkpoint.

This module provides file-based locking that works across processes.

Usage:
    from grail.infrastructure.checkpoint_lock import CheckpointLock

    async with CheckpointLock(cache_root, window=12345) as lock:
        if lock.should_download:
            # This worker is responsible for downloading
            await download_checkpoint(window)
            lock.mark_complete()
        else:
            # Another worker is downloading, wait for it
            await lock.wait_for_download()

    # Checkpoint is now available at cache_root / f"checkpoint-{window}"
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

# Lock file timeout (seconds) - if a lock is older than this, consider it stale
LOCK_TIMEOUT_SECONDS = 600  # 10 minutes (enough for large model downloads)

# Poll interval when waiting for another worker's download
POLL_INTERVAL_SECONDS = 2.0

# Maximum time to wait for another worker's download
MAX_WAIT_SECONDS = 900  # 15 minutes


class CheckpointLock:
    """Cross-process lock for checkpoint downloads.

    Uses file-based locking to coordinate checkpoint downloads across
    multiple worker processes. Only one worker downloads; others wait.

    Attributes:
        cache_root: Root directory for checkpoint cache
        window: Checkpoint window number
        should_download: True if this worker should perform the download
    """

    def __init__(self, cache_root: Path, window: int) -> None:
        self.cache_root = Path(cache_root)
        self.window = window
        self.should_download = False

        # Lock file path
        self._lock_dir = self.cache_root / ".locks"
        self._lock_file = self._lock_dir / f"checkpoint-{window}.lock"
        self._complete_file = self._lock_dir / f"checkpoint-{window}.complete"

        # Worker identification
        self._worker_id = os.getenv("GRAIL_WORKER_ID", "0")
        self._pid = os.getpid()

    async def __aenter__(self) -> "CheckpointLock":
        """Acquire lock or determine if should wait."""
        self._lock_dir.mkdir(parents=True, exist_ok=True)

        # Check if checkpoint already exists and is valid
        checkpoint_dir = self.cache_root / f"checkpoint-{self.window}"
        if checkpoint_dir.exists() and self._is_checkpoint_valid(checkpoint_dir):
            logger.debug(
                "Checkpoint %s already cached, no download needed (worker %s)",
                self.window,
                self._worker_id,
            )
            self.should_download = False
            return self

        # Check if download is already complete (another worker finished)
        if self._complete_file.exists():
            logger.debug(
                "Checkpoint %s download already complete (worker %s)",
                self.window,
                self._worker_id,
            )
            self.should_download = False
            return self

        # Try to acquire the lock
        try:
            self.should_download = await self._try_acquire_lock()
        except Exception as e:
            logger.warning("Failed to acquire checkpoint lock: %s", e)
            # Fall back to downloading (might duplicate work but won't deadlock)
            self.should_download = True

        if self.should_download:
            logger.info(
                "Worker %s (PID %s) acquired lock for checkpoint %s download",
                self._worker_id,
                self._pid,
                self.window,
            )
        else:
            logger.info(
                "Worker %s waiting for checkpoint %s (another worker downloading)",
                self._worker_id,
                self.window,
            )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release lock on exit."""
        if self.should_download:
            # Clean up lock file (but keep complete marker)
            try:
                if self._lock_file.exists():
                    self._lock_file.unlink()
            except Exception as e:
                logger.debug("Failed to remove lock file: %s", e)

    async def _try_acquire_lock(self) -> bool:
        """Attempt to acquire the lock file atomically.

        Returns:
            True if lock acquired (this worker should download),
            False if another worker holds the lock
        """
        # Check for stale lock
        if self._lock_file.exists():
            try:
                lock_age = time.time() - self._lock_file.stat().st_mtime
                if lock_age > LOCK_TIMEOUT_SECONDS:
                    logger.warning(
                        "Stale lock detected for checkpoint %s (age: %.0fs), removing",
                        self.window,
                        lock_age,
                    )
                    self._lock_file.unlink()
                else:
                    # Lock exists and is recent - another worker is downloading
                    return False
            except FileNotFoundError:
                pass  # Lock was removed between check and unlink

        # Try to create lock file atomically
        try:
            # Use os.open with O_CREAT | O_EXCL for atomic creation
            fd = os.open(
                str(self._lock_file),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            # Write worker info for debugging
            lock_info = f"worker={self._worker_id}\npid={self._pid}\ntime={time.time()}\n"
            os.write(fd, lock_info.encode())
            os.close(fd)
            return True
        except FileExistsError:
            # Another worker created the lock first
            return False

    async def wait_for_download(self, timeout: float = MAX_WAIT_SECONDS) -> bool:
        """Wait for another worker to complete the checkpoint download.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if download completed, False if timeout or error
        """
        start_time = time.time()
        checkpoint_dir = self.cache_root / f"checkpoint-{self.window}"

        while time.time() - start_time < timeout:
            # Check if download completed
            if self._complete_file.exists() or (
                checkpoint_dir.exists() and self._is_checkpoint_valid(checkpoint_dir)
            ):
                elapsed = time.time() - start_time
                logger.info(
                    "Checkpoint %s ready after %.1fs wait (worker %s)",
                    self.window,
                    elapsed,
                    self._worker_id,
                )
                return True

            # Check if lock was released without completion (download failed)
            if not self._lock_file.exists() and not self._complete_file.exists():
                logger.warning(
                    "Lock released without completion for checkpoint %s, will retry",
                    self.window,
                )
                return False

            # Wait and poll again
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        logger.error(
            "Timeout waiting for checkpoint %s download (waited %.0fs)",
            self.window,
            timeout,
        )
        return False

    def mark_complete(self) -> None:
        """Mark the download as complete for other workers."""
        try:
            self._complete_file.write_text(
                f"window={self.window}\nworker={self._worker_id}\n"
                f"pid={self._pid}\ntime={time.time()}\n"
            )
            logger.debug(
                "Marked checkpoint %s download complete (worker %s)",
                self.window,
                self._worker_id,
            )
        except Exception as e:
            logger.warning("Failed to mark download complete: %s", e)

    def _is_checkpoint_valid(self, checkpoint_dir: Path) -> bool:
        """Quick check if checkpoint directory has expected files."""
        # Check for essential files
        has_model = (
            (checkpoint_dir / "model.safetensors").exists()
            or any(checkpoint_dir.glob("model-*.safetensors"))
        )
        has_config = (checkpoint_dir / "config.json").exists()
        return has_model and has_config


def cleanup_stale_locks(cache_root: Path, max_age_seconds: float = LOCK_TIMEOUT_SECONDS) -> int:
    """Remove stale lock files older than max_age_seconds.

    Args:
        cache_root: Checkpoint cache root directory
        max_age_seconds: Maximum age before considering a lock stale

    Returns:
        Number of stale locks removed
    """
    lock_dir = cache_root / ".locks"
    if not lock_dir.exists():
        return 0

    removed = 0
    current_time = time.time()

    for lock_file in lock_dir.glob("*.lock"):
        try:
            lock_age = current_time - lock_file.stat().st_mtime
            if lock_age > max_age_seconds:
                lock_file.unlink()
                removed += 1
                logger.info("Removed stale lock: %s (age: %.0fs)", lock_file.name, lock_age)
        except Exception as e:
            logger.debug("Failed to check/remove lock %s: %s", lock_file, e)

    return removed
