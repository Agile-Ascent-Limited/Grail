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

# Timeout for claim recycling - if a problem is claimed but not completed
# within this time, the hub can recycle it for another worker to claim
# Default 45s - accommodates cross-datacenter Redis latency + generation time
CLAIM_TIMEOUT_SECONDS = float(os.getenv("GRAIL_CLAIM_TIMEOUT", "45"))

# Shared node ID key PREFIX - actual key is per-hostname to prevent cross-node collision
# Each physical node uses: grail:config:node_id:{hostname}
REDIS_NODE_ID_KEY_PREFIX = "grail:config:node_id"
REDIS_NODE_ID_TTL = 86400  # 24 hours


def _get_node_id_key() -> str:
    """Get hostname-specific Redis key for node_id sharing."""
    import socket
    hostname = socket.gethostname()
    return f"{REDIS_NODE_ID_KEY_PREFIX}:{hostname}"

# Maximum problems per window (configurable via env var)
# Default 2000 supports up to 4 nodes × 8 GPUs at ~500 rollouts/worker
# (each problem = 8 rollouts with completion_n=8)
MAX_PROBLEMS_PER_WINDOW = int(os.getenv("GRAIL_MAX_PROBLEMS_PER_WINDOW", "2000"))


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

    def mark_completed(self, window: int, problem_index: int) -> None:
        """Mark a problem as completed (worker finished generating rollouts)."""
        ...

    def recycle_stale_claims(self, window: int, timeout: float | None = None) -> int:
        """Hub-only: Find stale claims and recycle them. Returns count recycled."""
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
            server_id: Unique identifier for this server (shared via Redis if None)
        """
        import redis

        self.redis_url = redis_url
        self.worker_id = worker_id
        self.is_leader = worker_id == 0

        # Connect to Redis first (needed for server_id sharing)
        self.client = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis: %s", e)
            raise

        # Get or share server_id via Redis (ensures all workers use same ID)
        self.server_id = self._get_or_share_server_id(server_id)

        logger.info(
            "Redis problem queue connected: %s (server=%s, worker=%d)",
            self._safe_url(redis_url),
            self.server_id,
            worker_id,
        )

    def _generate_server_id(self) -> str:
        """Generate a unique server identifier."""
        import socket
        import uuid

        hostname = socket.gethostname()
        short_uuid = str(uuid.uuid4())[:8]
        return f"{hostname}-{short_uuid}"

    def _get_or_share_server_id(self, provided_id: str | None) -> str:
        """
        Get shared server_id from Redis or set it if worker 0.

        This ensures all workers on the same node use the same server_id,
        giving consistent claim keys like node-1:worker0, node-1:worker1.

        Args:
            provided_id: Server ID from environment variable (if any)

        Returns:
            The server_id to use (shared across all workers)
        """
        # Get hostname-specific key for node_id sharing (prevents cross-node collision)
        node_id_key = _get_node_id_key()

        # If explicitly provided via env var, use it
        if provided_id:
            if self.is_leader:
                # Leader stores the provided server_id in Redis for other workers on same host
                try:
                    self.client.setex(node_id_key, REDIS_NODE_ID_TTL, provided_id)
                    logger.info(
                        "Worker 0: stored server_id '%s' in Redis key '%s' for other workers",
                        provided_id, node_id_key,
                    )
                except Exception as e:
                    logger.warning("Failed to store server_id in Redis: %s", e)
            return provided_id

        # No provided server_id - use Redis for coordination
        if self.is_leader:
            # Leader generates and stores a server_id
            generated_id = self._generate_server_id()
            try:
                # Use SETNX to avoid overwriting if another leader already set it
                set_result = self.client.setnx(node_id_key, generated_id)
                if set_result:
                    self.client.expire(node_id_key, REDIS_NODE_ID_TTL)
                    logger.info(
                        "Worker 0: generated and stored server_id '%s' in Redis key '%s'",
                        generated_id, node_id_key,
                    )
                    return generated_id
                else:
                    # Another process already set it, read and use that one
                    existing_id = self.client.get(node_id_key)
                    if existing_id:
                        logger.info(
                            "Worker 0: using existing server_id '%s' from Redis",
                            existing_id,
                        )
                        return existing_id
                    return generated_id
            except Exception as e:
                logger.warning("Failed to coordinate server_id via Redis: %s", e)
                return generated_id
        else:
            # Follower tries to read from Redis
            try:
                # Wait a bit for leader to set it (up to 5 seconds)
                for _ in range(50):  # 50 * 0.1s = 5 seconds
                    shared_id = self.client.get(node_id_key)
                    if shared_id:
                        logger.info(
                            "Worker %d: using shared server_id '%s' from Redis",
                            self.worker_id,
                            shared_id,
                        )
                        return shared_id
                    time.sleep(0.1)

                # Timeout - generate our own (fallback)
                logger.warning(
                    "Worker %d: no shared server_id in Redis after 5s, generating own",
                    self.worker_id,
                )
                return self._generate_server_id()
            except Exception as e:
                logger.warning("Failed to read server_id from Redis: %s", e)
                return self._generate_server_id()

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

    def _complete_key(self, window: int, problem_index: int) -> str:
        """Get Redis key for problem completion marker."""
        return f"{REDIS_PREFIX}:{window}:complete:{problem_index}"

    def _recycle_queue_key(self, window: int) -> str:
        """Get Redis key for recycle queue (list of problem indices to re-claim)."""
        return f"{REDIS_PREFIX}:{window}:recycle"

    def claim_next_problem(self, window: int, max_attempts: int = MAX_PROBLEMS_PER_WINDOW) -> int:
        """
        Atomically claim the next problem index.

        First checks the recycle queue for timed-out problems that can be re-claimed.
        If no recycled problems available, uses Redis INCR for a new index.

        Args:
            window: Window start block number
            max_attempts: Maximum problem index (safety limit, default from GRAIL_MAX_PROBLEMS_PER_WINDOW)

        Returns:
            The claimed problem index, or -1 on error
        """
        try:
            # First, try to claim from recycle queue (timed-out problems)
            recycle_key = self._recycle_queue_key(window)
            recycled = self.client.lpop(recycle_key)
            if recycled is not None:
                problem_index = int(recycled)
                # Record the re-claim
                claim_key = self._claim_key(window, problem_index)
                self.client.setex(
                    claim_key,
                    CLAIM_TTL,
                    f"{self.server_id}:worker{self.worker_id}:{time.time()}:recycled"
                )
                logger.info(
                    "Worker %d@%s re-claimed recycled problem %d for window %d",
                    self.worker_id,
                    self.server_id,
                    problem_index,
                    window,
                )
                return problem_index

            # No recycled problems, claim a new one
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

    def mark_completed(self, window: int, problem_index: int) -> None:
        """
        Mark a problem as completed (worker finished generating rollouts).

        This is used by the hub to detect stale claims (claimed but not completed).

        Args:
            window: Window start block number
            problem_index: The problem index that was completed
        """
        try:
            complete_key = self._complete_key(window, problem_index)
            self.client.setex(
                complete_key,
                CLAIM_TTL,
                f"{self.server_id}:worker{self.worker_id}:{time.time()}"
            )
            logger.debug(
                "Worker %d@%s marked problem %d complete for window %d",
                self.worker_id,
                self.server_id,
                problem_index,
                window,
            )
        except Exception as e:
            logger.warning("Redis mark_completed failed: %s", e)

    def recycle_stale_claims(self, window: int, timeout: float | None = None) -> int:
        """
        Hub-only: Find stale claims (claimed but not completed) and recycle them.

        Scans all claim keys for this window, checks if they have a corresponding
        completion key, and if not and the claim is older than timeout, adds the
        problem index to the recycle queue.

        Args:
            window: Window start block number
            timeout: Seconds before a claim is considered stale (default: CLAIM_TIMEOUT_SECONDS)

        Returns:
            Number of problems recycled
        """
        if timeout is None:
            timeout = CLAIM_TIMEOUT_SECONDS

        recycled_count = 0
        now = time.time()

        try:
            # Scan for all claim keys
            claim_pattern = f"{REDIS_PREFIX}:{window}:claim:*"
            cursor = 0

            while True:
                cursor, keys = self.client.scan(cursor, match=claim_pattern, count=100)

                for claim_key in keys:
                    # Extract problem index from key
                    try:
                        problem_index = int(claim_key.split(":")[-1])
                    except ValueError:
                        continue

                    # Check if already completed
                    complete_key = self._complete_key(window, problem_index)
                    if self.client.exists(complete_key):
                        continue  # Already completed, skip

                    # Check claim age
                    claim_value = self.client.get(claim_key)
                    if not claim_value:
                        continue

                    # Parse timestamp from claim value (format: "server:workerN:timestamp[:recycled]")
                    parts = claim_value.split(":")
                    if len(parts) < 3:
                        continue

                    try:
                        claim_time = float(parts[2])
                    except ValueError:
                        continue

                    claim_age = now - claim_time
                    if claim_age > timeout:
                        # Stale claim - add to recycle queue
                        recycle_key = self._recycle_queue_key(window)
                        # Use RPUSH to add to end of queue (FIFO)
                        self.client.rpush(recycle_key, str(problem_index))
                        self.client.expire(recycle_key, CLAIM_TTL)

                        # Delete the stale claim to prevent re-recycling
                        self.client.delete(claim_key)

                        recycled_count += 1
                        logger.info(
                            "🔄 Hub recycled stale problem %d for window %d "
                            "(claimed %.1fs ago by %s, not completed)",
                            problem_index,
                            window,
                            claim_age,
                            ":".join(parts[:2]),
                        )

                if cursor == 0:
                    break

            if recycled_count > 0:
                logger.info(
                    "🔄 Hub recycled %d stale claims for window %d (timeout=%.0fs)",
                    recycled_count,
                    window,
                    timeout,
                )

        except Exception as e:
            logger.warning("Redis recycle_stale_claims failed: %s", e)

        return recycled_count

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
    _cache_root,  # Unused, kept for API compatibility
    worker_id: int,
    total_workers: int = 8,
) -> ProblemQueueProtocol:
    """
    Get Redis-based problem queue for cross-server coordination.

    GRAIL_REDIS_URL is required for mining to ensure unique problem indices
    across all workers and nodes.

    Args:
        _cache_root: Unused, kept for API compatibility
        worker_id: This worker's ID
        total_workers: Total workers (unused)

    Returns:
        RedisProblemQueue instance

    Raises:
        RuntimeError: If GRAIL_REDIS_URL is not set or Redis connection fails
    """
    redis_url = os.getenv("GRAIL_REDIS_URL", "").strip()

    if not redis_url:
        raise RuntimeError(
            "GRAIL_REDIS_URL environment variable is required for mining. "
            "Set it to your Redis server URL (e.g., redis://localhost:6379/0)"
        )

    try:
        # Import redis to check if available
        import redis  # noqa: F401

        # Check GRAIL_NODE_ID first, then GRAIL_SERVER_ID for backwards compatibility
        server_id = os.getenv("GRAIL_NODE_ID") or os.getenv("GRAIL_SERVER_ID")
        return RedisProblemQueue(redis_url, worker_id, server_id)

    except ImportError:
        raise RuntimeError(
            "GRAIL_REDIS_URL set but redis package not installed. "
            "Install with: pip install redis"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Redis at {redis_url}: {e}")
