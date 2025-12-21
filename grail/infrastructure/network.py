from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, ClassVar

import bittensor as bt

from ..logging_utils import await_with_stall_log

logger = logging.getLogger(__name__)

# Timeout constants for subtensor lifecycle operations
# These prevent indefinite hangs during connection management
SUBTENSOR_CLOSE_TIMEOUT = float(os.getenv("BT_CLOSE_TIMEOUT", "30.0"))
SUBTENSOR_INIT_TIMEOUT = float(os.getenv("BT_INIT_TIMEOUT", "120.0"))


class ResilientSubtensor:
    """
    Wrapper around bittensor subtensor with circuit breaker, caching, and auto-restart.

    This prevents the application from hanging indefinitely when blockchain RPC
    calls fail or timeout. Protected methods will retry with exponential backoff.
    - Circuit breaker: Stops calls temporarily after repeated failures
    - Metagraph cache: Returns last good metagraph when calls timeout
    - Auto-restart: Recreates subtensor connection after extended failures
    """

    # Methods that should be protected with timeout and retry logic
    PROTECTED_METHODS: ClassVar[set[str]] = {
        "get_current_block",
        "get_block_hash",
        "metagraph",
        "get_commitment",
    }

    def __init__(
        self,
        subtensor: bt.subtensor,
        timeout: float = 10.0,
        retries: int = 3,
        backoff_base: float = 10.0,
    ):
        """
        Initialize resilient subtensor wrapper.

        Args:
            subtensor: The underlying bittensor subtensor instance
            timeout: Timeout in seconds for each attempt (default: 15s)
            retries: Number of retry attempts (default: 3)
            backoff_base: Base multiplier for exponential backoff (default: 5s)
        """
        object.__setattr__(self, "_subtensor", subtensor)
        object.__setattr__(self, "_timeout", timeout)
        object.__setattr__(self, "_retries", retries)
        object.__setattr__(self, "_backoff_base", backoff_base)
        # Circuit breaker state
        object.__setattr__(self, "_failure_count", 0)
        object.__setattr__(self, "_circuit_open_until", 0.0)
        object.__setattr__(self, "_circuit_threshold", 2)  # Open after 2 consecutive call failures
        object.__setattr__(self, "_circuit_timeout", 30.0)  # Stay open for 30s
        # Metagraph cache
        object.__setattr__(self, "_metagraph_cache", {})
        # Track last successful call for idle detection
        object.__setattr__(self, "_last_call_timestamp", time.time())

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        circuit_open_until = object.__getattribute__(self, "_circuit_open_until")
        return time.time() < circuit_open_until

    def _get_cached_metagraph(self, netuid: int) -> Any | None:
        """Get cached metagraph for a given netuid."""
        metagraph_cache = object.__getattribute__(self, "_metagraph_cache")
        return metagraph_cache.get(netuid)

    def _cache_metagraph(self, netuid: int, metagraph: Any) -> None:
        """Cache metagraph for a given netuid."""
        metagraph_cache = object.__getattribute__(self, "_metagraph_cache")
        metagraph_cache[netuid] = metagraph

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker failure count on success."""
        object.__setattr__(self, "_failure_count", 0)

    def _increment_failure_count(self) -> int:
        """Increment and return failure count."""
        failure_count = object.__getattribute__(self, "_failure_count") + 1
        object.__setattr__(self, "_failure_count", failure_count)
        return failure_count

    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker for cooldown period."""
        circuit_timeout = object.__getattribute__(self, "_circuit_timeout")
        failure_count = object.__getattribute__(self, "_failure_count")
        object.__setattr__(self, "_circuit_open_until", time.time() + circuit_timeout)
        logger.error(
            "🔴 Circuit breaker opened after %d failures, cooling down for %ds",
            failure_count,
            circuit_timeout,
        )

    def _should_open_circuit(self) -> bool:
        """Check if circuit breaker threshold is reached."""
        failure_count = object.__getattribute__(self, "_failure_count")
        circuit_threshold = object.__getattribute__(self, "_circuit_threshold")
        return failure_count >= circuit_threshold

    async def _restart_subtensor(self) -> None:
        """Restart subtensor connection with timeouts to prevent indefinite hangs.

        This method is called when:
        1. Connection has been idle for >60s (stale WebSocket)
        2. Circuit breaker opens after repeated failures

        Both close() and initialize() operations have hard timeouts to ensure
        the event loop is never blocked indefinitely, which would prevent
        heartbeat callbacks and trigger watchdog shutdown.

        Raises:
            asyncio.TimeoutError: If initialization times out after SUBTENSOR_INIT_TIMEOUT
        """
        logger.warning("🔄 Restarting subtensor connection...")
        subtensor = object.__getattribute__(self, "_subtensor")
        network = (
            subtensor.network
            if hasattr(subtensor, "network")
            else os.getenv("BT_NETWORK", "finney")
        )

        # Close old subtensor with timeout to prevent resource leaks
        # If close hangs (e.g., half-open WebSocket), we continue anyway
        try:
            if hasattr(subtensor, "close"):
                if asyncio.iscoroutinefunction(subtensor.close):
                    try:
                        await asyncio.wait_for(
                            subtensor.close(),
                            timeout=SUBTENSOR_CLOSE_TIMEOUT,
                        )
                        logger.debug("Closed old subtensor connection")
                    except asyncio.TimeoutError:
                        logger.warning(
                            "⏱️ Timeout closing old subtensor after %.0fs, continuing anyway",
                            SUBTENSOR_CLOSE_TIMEOUT,
                        )
                    except asyncio.CancelledError:
                        # Propagate real cancellation
                        raise
                else:
                    subtensor.close()
                    logger.debug("Closed old subtensor connection (sync)")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Failed to close old subtensor: %s", exc)

        # Create and initialize new subtensor with timeout
        # If this times out, we raise to let caller handle it
        new_subtensor = bt.async_subtensor(network=network)
        try:
            await asyncio.wait_for(
                new_subtensor.initialize(),
                timeout=SUBTENSOR_INIT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                "❌ Timeout initializing new subtensor after %.0fs",
                SUBTENSOR_INIT_TIMEOUT,
            )
            # Try to close the failed connection to prevent resource leak
            try:
                if hasattr(new_subtensor, "close"):
                    if asyncio.iscoroutinefunction(new_subtensor.close):
                        # Short timeout for cleanup
                        await asyncio.wait_for(new_subtensor.close(), timeout=5.0)
                    else:
                        new_subtensor.close()
            except Exception:
                pass
            raise
        except asyncio.CancelledError:
            # Clean up on cancellation
            try:
                if hasattr(new_subtensor, "close"):
                    if asyncio.iscoroutinefunction(new_subtensor.close):
                        await asyncio.wait_for(new_subtensor.close(), timeout=5.0)
                    else:
                        new_subtensor.close()
            except Exception:
                pass
            raise

        object.__setattr__(self, "_subtensor", new_subtensor)
        # Update last call timestamp to prevent immediate idle detection
        object.__setattr__(self, "_last_call_timestamp", time.time())
        self._reset_circuit_breaker()
        logger.info("✅ Subtensor connection restarted")

    async def restart(self) -> None:
        """Public method to restart subtensor connection."""
        await self._restart_subtensor()

    async def close(self) -> None:
        """Close the underlying subtensor connection with timeout."""
        subtensor = object.__getattribute__(self, "_subtensor")
        if hasattr(subtensor, "close"):
            if asyncio.iscoroutinefunction(subtensor.close):
                try:
                    await asyncio.wait_for(
                        subtensor.close(),
                        timeout=SUBTENSOR_CLOSE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "⏱️ Timeout closing subtensor after %.0fs",
                        SUBTENSOR_CLOSE_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Error closing subtensor: %s", exc)
            else:
                subtensor.close()

    async def _handle_circuit_open(self, method_name: str, args: tuple) -> Any:
        """Handle method call when circuit breaker is open."""
        if method_name == "metagraph" and args:
            cached = self._get_cached_metagraph(args[0])
            if cached:
                logger.warning("⚡ Circuit open, returning cached metagraph for netuid %s", args[0])
                return cached
        logger.warning("⚡ Circuit breaker open, skipping %s call", method_name)
        raise TimeoutError(f"Circuit breaker open for {method_name}")

    async def _attempt_call(self, method: Any, args: tuple, kwargs: dict, timeout: float) -> Any:
        """Attempt a single method call with timeout."""
        return await asyncio.wait_for(method(*args, **kwargs), timeout=timeout)

    def _handle_success(self, method_name: str, args: tuple, result: Any, retry: int) -> None:
        """Handle successful method call."""
        self._reset_circuit_breaker()
        if method_name == "metagraph" and args:
            self._cache_metagraph(args[0], result)
        if retry > 0:
            logger.info("✅ %s() succeeded on attempt %d", method_name, retry + 1)
        # Update last call timestamp
        object.__setattr__(self, "_last_call_timestamp", time.time())

    async def _handle_retry_backoff(
        self,
        reason: str,
        method_name: str,
        retry: int,
        retries: int,
        backoff_base: float,
    ) -> None:
        """Handle retry backoff for timeout or cancellation events."""
        if retry < retries - 1:
            wait_time = backoff_base * (2**retry)
            logger.error(
                "⏱️ %s in %s (attempt %d/%d), retrying in %ds",
                reason,
                method_name,
                retry + 1,
                retries,
                wait_time,
            )
            await asyncio.sleep(wait_time)

    def _handle_all_retries_failed(self, method_name: str, args: tuple, retries: int) -> Any:
        """Handle case when all retries are exhausted.

        Note: This counts complete call failures (after all retries), not individual
        retry attempts. Circuit opens after threshold consecutive call failures.
        """
        self._increment_failure_count()

        if self._should_open_circuit():
            self._open_circuit_breaker()
            # Schedule restart as background task with error handling
            restart_task = asyncio.create_task(
                self._restart_subtensor(),
                name="subtensor_restart",
            )
            # Add callback to log any errors from the background restart
            restart_task.add_done_callback(self._on_restart_complete)

        # Try to return cached metagraph as last resort
        if method_name == "metagraph" and args:
            cached = self._get_cached_metagraph(args[0])
            if cached:
                logger.warning("⚠️ Returning stale cached metagraph for netuid %s", args[0])
                return cached

        logger.error("❌ %s failed after %d attempts", method_name, retries)
        raise TimeoutError(f"{method_name} failed after {retries} attempts")

    def _on_restart_complete(self, task: asyncio.Task) -> None:
        """Callback for background restart task completion."""
        try:
            # Check if task raised an exception
            exc = task.exception()
            if exc is not None:
                logger.error("⚠️ Background subtensor restart failed: %s", exc)
        except asyncio.CancelledError:
            logger.debug("Background subtensor restart was cancelled")
        except asyncio.InvalidStateError:
            # Task not done yet (shouldn't happen in done callback)
            pass

    async def _call_with_retry(
        self, method_name: str, method: Any, args: tuple, kwargs: dict
    ) -> Any:
        """Execute method with retry logic and circuit breaker protection."""
        # Check circuit breaker
        if self._is_circuit_open():
            return await self._handle_circuit_open(method_name, args)

        timeout = object.__getattribute__(self, "_timeout")
        retries = object.__getattribute__(self, "_retries")
        backoff_base = object.__getattribute__(self, "_backoff_base")

        # Double timeout for metagraph calls
        if method_name == "metagraph":
            timeout = timeout * 2

        # Restart connection if it has been idle for too long
        # Research-based threshold:
        # - Bittensor WebSocket auto-closes after 10s inactivity
        # - Substrate layer closes after ~60s inactivity
        # - 60s catches stale connections while avoiding false positives
        # - Critical for upload worker (40-300s idle during R2 uploads)
        last_call_timestamp = object.__getattribute__(self, "_last_call_timestamp")
        idle_duration = time.time() - last_call_timestamp
        if idle_duration > 60.0:
            logger.warning(
                "⏰ Connection idle for %.1fs, restarting subtensor and doubling timeout for %s",
                idle_duration,
                method_name,
            )
            # Update timestamp BEFORE restart attempt to prevent retry loops
            # Even if restart fails, we don't want to immediately retry
            object.__setattr__(self, "_last_call_timestamp", time.time())
            try:
                await self._restart_subtensor()
            except asyncio.TimeoutError:
                # Restart timed out - double the timeout and try the call anyway
                # The old connection might still work, or we'll fail fast
                logger.warning(
                    "⚠️ Subtensor restart timed out, attempting call with doubled timeout"
                )
                timeout = timeout * 2
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "⚠️ Subtensor restart failed (%s), attempting call anyway", exc
                )
                timeout = timeout * 2

        # Retry loop
        for retry in range(retries):
            try:
                result = await self._attempt_call(method, args, kwargs, timeout)
                self._handle_success(method_name, args, result, retry)
                return result
            except asyncio.TimeoutError:
                await self._handle_retry_backoff(
                    "Timeout",
                    method_name,
                    retry,
                    retries,
                    backoff_base,
                )
            except asyncio.CancelledError:
                # If our task is actually being cancelled, propagate.
                task = asyncio.current_task()
                cancelling_count = 0
                if task is not None and hasattr(task, "cancelling"):
                    try:
                        # Python 3.11+: number of cancellation requests
                        cancelling_count = task.cancelling()  # type: ignore[attr-defined]
                    except Exception:
                        cancelling_count = 0
                if cancelling_count > 0:
                    raise
                # Otherwise, treat as transient cancellation from underlying client
                await self._handle_retry_backoff(
                    "Cancellation",
                    method_name,
                    retry,
                    retries,
                    backoff_base,
                )

        # All retries exhausted
        return self._handle_all_retries_failed(method_name, args, retries)

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access to wrap protected methods."""
        attr = getattr(object.__getattribute__(self, "_subtensor"), name)

        # Only wrap methods we want to protect
        if name not in self.PROTECTED_METHODS or not callable(attr):
            return attr

        # Check if it's an async method
        if not asyncio.iscoroutinefunction(attr):
            return attr

        # Return a wrapped version with retry logic
        async def wrapped_method(*args: Any, **kwargs: Any) -> Any:
            return await self._call_with_retry(name, attr, args, kwargs)

        return wrapped_method

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to underlying subtensor."""
        setattr(object.__getattribute__(self, "_subtensor"), name, value)

    def __repr__(self) -> str:
        """String representation of the resilient subtensor."""
        subtensor = object.__getattribute__(self, "_subtensor")
        timeout = object.__getattribute__(self, "_timeout")
        retries = object.__getattribute__(self, "_retries")
        return f"ResilientSubtensor(subtensor={subtensor}, timeout={timeout}s, retries={retries})"


def _resolve_network() -> tuple[str, str | None]:
    """
    Resolve network selection from environment variables.

    Priority:
    - BT_CHAIN_ENDPOINT: if set, use custom endpoint (overrides BT_NETWORK)
    - BT_NETWORK: named network ('finney', 'test', 'local'), defaults to 'finney'

    Returns:
        (network, chain_endpoint): network name and optional custom endpoint
    """
    network = os.getenv("BT_NETWORK", "finney")
    chain_endpoint = os.getenv("BT_CHAIN_ENDPOINT")  # No default - None if unset
    return network, chain_endpoint


async def create_subtensor(*, resilient: bool = True) -> bt.subtensor | ResilientSubtensor:
    """
    Create and initialize an async subtensor instance using env configuration.

    - If BT_CHAIN_ENDPOINT is set, connect to custom endpoint directly
    - Otherwise, use BT_NETWORK ('finney', 'test', 'local') - defaults to 'finney'

    The Bittensor SDK resolves named networks to official endpoints automatically.

    Args:
        resilient: If True, wrap subtensor with ResilientSubtensor for automatic
                   timeout and retry logic (default: True, recommended for production)

    Environment Variables:
        BT_NETWORK: Network name ('finney', 'test', 'local')
        BT_CHAIN_ENDPOINT: Custom WebSocket endpoint URL
        BT_CALL_TIMEOUT: Timeout in seconds for blockchain calls (default: 15.0)
        BT_CALL_RETRIES: Number of retry attempts (default: 3)
        BT_CALL_BACKOFF: Base backoff multiplier in seconds (default: 5.0)

    Returns:
        Initialized subtensor instance (optionally wrapped with resilience layer)
    """
    network, chain_endpoint = _resolve_network()

    if chain_endpoint:
        # Custom endpoint specified (e.g., local node, custom remote)
        logger.info(
            "Connecting to custom Bittensor endpoint: %s (BT_NETWORK=%s ignored)",
            chain_endpoint,
            network,
        )
        subtensor = bt.async_subtensor(network=chain_endpoint)
    else:
        # Use named network - SDK resolves to official endpoint
        label = {
            "finney": "mainnet",
            "test": "testnet",
            "local": "local",
        }.get(network, "custom")

        if network not in {"finney", "test", "local"}:
            logger.warning(
                "Unknown BT_NETWORK='%s', defaulting to 'finney'. "
                "Valid options: finney, test, local",
                network,
            )
            network = "finney"
            label = "mainnet"

        logger.info("Connecting to Bittensor %s (network=%s)", label, network)
        subtensor = bt.async_subtensor(network=network)

    # Initialize with hard timeout to prevent indefinite hangs
    # await_with_stall_log only warns, so we wrap with wait_for for hard timeout
    try:
        await asyncio.wait_for(
            await_with_stall_log(
                subtensor.initialize(),
                label="subtensor.initialize",
                threshold_seconds=60.0,  # Warn after 60s
                log=logger,
            ),
            timeout=SUBTENSOR_INIT_TIMEOUT,  # Hard timeout (default 120s)
        )
    except asyncio.TimeoutError:
        logger.error(
            "❌ Subtensor initialization timed out after %.0fs",
            SUBTENSOR_INIT_TIMEOUT,
        )
        # Clean up the failed connection
        try:
            if hasattr(subtensor, "close"):
                if asyncio.iscoroutinefunction(subtensor.close):
                    await asyncio.wait_for(subtensor.close(), timeout=5.0)
                else:
                    subtensor.close()
        except Exception:
            pass
        raise

    if resilient:
        # Wrap with resilience layer for production use
        timeout = float(os.getenv("BT_CALL_TIMEOUT", "15.0"))
        retries = int(os.getenv("BT_CALL_RETRIES", "3"))
        backoff = float(os.getenv("BT_CALL_BACKOFF", "5.0"))

        logger.info(
            "Wrapping subtensor with resilience layer (timeout=%ds, retries=%d, backoff=%ds)",
            timeout,
            retries,
            backoff,
        )
        return ResilientSubtensor(subtensor, timeout=timeout, retries=retries, backoff_base=backoff)

    return subtensor
