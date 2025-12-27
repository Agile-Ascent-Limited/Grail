"""
Redis-based problem queue for cross-server coordination.

Enables multiple servers running the same hotkey to coordinate which problems
they claim, avoiding duplicate work. Uses Redis INCR for atomic counter.

Usage:
    # Set environment variable to enable
    export GRAIL_REDIS_URL=redis://localhost:6379/0

    # Or with authentication
    export GRAIL_REDIS_URL=redis://:password@redis-host:6379/0

If GRAIL_REDIS_URL is not set, falls back to file-based ProblemQueue.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol

logger = logging.getLogger(__name__)

# Redis key prefix for all grail keys
REDIS_PREFIX = "grail:problems"

# How long to keep problem claims in Redis (seconds)
# Should be longer than a window (~6 minutes) but not too long
CLAIM_TTL = 600  # 10 minutes

# Maximum problems per window (configurable via env var)
# Default 500 supports ~8 H200s at 2 rollouts/s for 300s windows
MAX_PROBLEMS_PER_WINDOW = int(os.getenv("GRAIL_MAX_PROBLEMS_PER_WINDOW", "500"))


class ProblemQueueProtocol(Protocol):
    """Protocol for problem queue implementations."""

    def claim_next_problem(self, window: int, max_attempts: int = MAX_PROBLEMS_PER_WINDOW) -> int:
        """Claim the next available problem index."""
        ...

    def reset_counter(self, window: int) -> None:
        """Reset counter for a new window (leader only)."""
        ...

    def cleanup_window(self, window: int) -> None:
        """Clean up data for a completed window."""
        ...

    def get_current_count(self, window: int) -> int:
        """Get current problem count for monitoring."""
        ...

    def wait_for_leader_reset(self, window: int, timeout: float = 10.0) -> bool:
        """Wait for leader to reset (for compatibility)."""
        ...


class RedisProblemQueue:
    """
    Redis-based problem queue for cross-server coordination.

    Uses Redis INCR for atomic counter increment across all servers.
    Much simpler than file-based approach and works across machines.

    Key structure:
        grail:problems:{window}:counter - Atomic counter (0, 1, 2, ...)
        grail:problems:{window}:ready - Ready marker from leader
    """

    def __init__(self, redis_url: str, worker_id: int, server_id: str | None = None):
        """
        Initialize Redis problem queue.

        Args:
            redis_url: Redis connection URL (redis://host:port/db)
            worker_id: This worker's ID (0 = leader within a server)
            server_id: Unique identifier for this server (auto-generated if None)
        """
        import redis

        self.redis_url = redis_url
        self.worker_id = worker_id
        self.is_leader = worker_id == 0
        self.server_id = server_id or self._generate_server_id()

        # Connect to Redis
        self.client = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        try:
            self.client.ping()
            logger.info(
                "Redis problem queue connected: %s (server=%s, worker=%d)",
                self._safe_url(redis_url),
                self.server_id,
                worker_id,
            )
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis: %s", e)
            raise

    def _generate_server_id(self) -> str:
        """Generate a unique server identifier."""
        import socket
        import uuid

        hostname = socket.gethostname()
        short_uuid = str(uuid.uuid4())[:8]
        return f"{hostname}-{short_uuid}"

    def _safe_url(self, url: str) -> str:
        """Mask password in URL for logging."""
        if "@" in url and ":" in url.split("@")[0]:
            # Has password, mask it
            parts = url.split("@")
            auth_parts = parts[0].split(":")
            return f"{auth_parts[0]}:****@{parts[1]}"
        return url

    def _counter_key(self, window: int) -> str:
        """Get Redis key for window counter."""
        return f"{REDIS_PREFIX}:{window}:counter"

    def _ready_key(self, window: int) -> str:
        """Get Redis key for ready marker."""
        return f"{REDIS_PREFIX}:{window}:ready"

    def _claim_key(self, window: int, problem_index: int) -> str:
        """Get Redis key for problem claim (for tracking/debugging)."""
        return f"{REDIS_PREFIX}:{window}:claim:{problem_index}"

    def claim_next_problem(self, window: int, max_attempts: int = MAX_PROBLEMS_PER_WINDOW) -> int:
        """
        Atomically claim the next problem index.

        Uses Redis INCR which is atomic across all clients.
        Returns the claimed index (0-based).

        Args:
            window: Window start block number
            max_attempts: Maximum problem index (safety limit, default from GRAIL_MAX_PROBLEMS_PER_WINDOW)

        Returns:
            The claimed problem index, or -1 on error
        """
        try:
            counter_key = self._counter_key(window)

            # INCR returns the new value after increment
            # We use INCRBY 0 first to initialize if needed, then INCR
            # Actually, just use INCR - it initializes to 0 and increments
            next_value = self.client.incr(counter_key)

            # INCR returns 1 on first call (after incrementing 0 to 1)
            # We want 0-based indices, so subtract 1
            problem_index = next_value - 1

            if problem_index >= max_attempts:
                logger.warning(
                    "Worker %d@%s: reached max problems (%d) for window %d",
                    self.worker_id,
                    self.server_id,
                    max_attempts,
                    window,
                )
                return -1

            # Set TTL on counter to auto-cleanup
            self.client.expire(counter_key, CLAIM_TTL)

            # Record claim for debugging (optional, with TTL)
            claim_key = self._claim_key(window, problem_index)
            self.client.setex(
                claim_key,
                CLAIM_TTL,
                f"{self.server_id}:worker{self.worker_id}:{time.time()}"
            )

            logger.debug(
                "Worker %d@%s claimed problem %d for window %d",
                self.worker_id,
                self.server_id,
                problem_index,
                window,
            )
            return problem_index

        except Exception as e:
            logger.error("Redis claim_next_problem failed: %s", e)
            return -1

    def reset_counter(self, window: int) -> None:
        """
        Reset counter for a new window.

        Only the global leader (first server's worker 0) should call this.
        In practice, multiple resets are idempotent (just sets to 0).
        """
        try:
            counter_key = self._counter_key(window)
            ready_key = self._ready_key(window)

            # Delete counter to reset (INCR on non-existent key starts at 1)
            # Actually, let's just set to 0 so INCR gives 1, which - 1 = 0
            self.client.delete(counter_key)

            # Set ready marker with TTL
            self.client.setex(ready_key, CLAIM_TTL, f"{self.server_id}:{time.time()}")

            logger.info(
                "Worker %d@%s reset counter for window %d",
                self.worker_id,
                self.server_id,
                window,
            )
        except Exception as e:
            logger.error("Redis reset_counter failed: %s", e)

    def wait_for_leader_reset(self, window: int, timeout: float = 10.0) -> bool:
        """
        Wait for ready marker from leader.

        With Redis, this is less critical since INCR is atomic.
        But we keep it for consistency with file-based implementation.
        """
        if self.is_leader:
            return True

        try:
            ready_key = self._ready_key(window)
            start_time = time.time()

            while time.time() - start_time < timeout:
                if self.client.exists(ready_key):
                    return True
                time.sleep(0.1)

            # Timeout - proceed anyway (Redis counter still works)
            logger.debug(
                "Worker %d@%s: no ready marker for window %d, proceeding",
                self.worker_id,
                self.server_id,
                window,
            )
            return True  # Redis INCR works even without reset

        except Exception as e:
            logger.warning("Redis wait_for_leader_reset failed: %s", e)
            return True  # Proceed anyway

    def get_current_count(self, window: int) -> int:
        """Get current problem count for monitoring."""
        try:
            counter_key = self._counter_key(window)
            value = self.client.get(counter_key)
            return int(value) if value else 0
        except Exception:
            return 0

    def cleanup_window(self, window: int) -> None:
        """Clean up Redis keys for a completed window."""
        try:
            # Use SCAN to find all keys for this window
            pattern = f"{REDIS_PREFIX}:{window}:*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            if deleted > 0:
                logger.debug("Cleaned up %d Redis keys for window %d", deleted, window)

        except Exception as e:
            logger.warning("Redis cleanup_window failed: %s", e)


def get_problem_queue(
    cache_root,
    worker_id: int,
    total_workers: int = 8,
) -> ProblemQueueProtocol:
    """
    Get the appropriate problem queue implementation.

    Priority order:
    1. Static allocation (GRAIL_STATIC_ALLOCATION=1) - zero overhead, pre-assigned ranges
    2. Redis (GRAIL_REDIS_URL) - for cross-server coordination
    3. File-based - default single-server mode

    Args:
        cache_root: Cache root directory (for file-based fallback)
        worker_id: This worker's ID
        total_workers: Total workers

    Returns:
        Problem queue instance
    """
    from pathlib import Path

    # Priority 1: Static allocation (DDP-style, zero overhead)
    static_enabled = os.getenv("GRAIL_STATIC_ALLOCATION", "").lower() in ("1", "true", "yes")
    if static_enabled:
        from grail.infrastructure.worker_barrier import StaticProblemAllocator, PROBLEMS_PER_WORKER

        logger.info(
            "Using STATIC allocation: %d problems/worker, worker %d/%d (zero overhead)",
            PROBLEMS_PER_WORKER,
            worker_id,
            total_workers,
        )
        return StaticProblemAllocator(worker_id, total_workers, PROBLEMS_PER_WORKER)

    # Priority 2: Redis (cross-server coordination)
    redis_url = os.getenv("GRAIL_REDIS_URL", "").strip()

    if redis_url:
        try:
            # Import redis to check if available
            import redis  # noqa: F401

            server_id = os.getenv("GRAIL_SERVER_ID")
            return RedisProblemQueue(redis_url, worker_id, server_id)

        except ImportError:
            logger.warning(
                "GRAIL_REDIS_URL set but redis package not installed. "
                "Install with: pip install redis"
            )
        except Exception as e:
            logger.warning("Failed to connect to Redis, falling back to file-based: %s", e)

    # Priority 3: File-based (default single-server)
    from grail.infrastructure.worker_barrier import ProblemQueue

    return ProblemQueue(Path(cache_root), worker_id)
