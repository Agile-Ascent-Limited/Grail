"""Process cleanup utilities for GRAIL miners.

Provides functions to kill zombie vLLM and grail processes that may be
blocking GPU memory or interfering with startup.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def kill_zombie_vllm_processes(grace_period: float = 2.0) -> int:
    """Kill any zombie vLLM processes that may be blocking GPU memory.

    Args:
        grace_period: Seconds to wait after sending kill signals

    Returns:
        Number of processes killed
    """
    killed = 0

    if _is_windows():
        # Windows: use taskkill
        patterns = ["vllm", "VLLM"]
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", f"*{pattern}*"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    killed += 1
            except Exception:
                pass
    else:
        # Linux/macOS: use pkill
        patterns = ["vllm", "VLLM::EngineCor"]
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # pkill returns 0 if processes were killed
                if result.returncode == 0:
                    killed += 1
            except FileNotFoundError:
                # pkill not available
                pass
            except Exception as e:
                logger.debug(f"Error killing {pattern}: {e}")

    if killed > 0:
        logger.info(f"Killed {killed} zombie vLLM process groups")
        time.sleep(grace_period)

    return killed


def kill_zombie_grail_processes(exclude_self: bool = True, grace_period: float = 1.0) -> int:
    """Kill any zombie grail processes from previous runs.

    Args:
        exclude_self: If True, don't kill the current process
        grace_period: Seconds to wait after sending kill signals

    Returns:
        Number of processes killed
    """
    killed = 0
    my_pid = os.getpid()

    if _is_windows():
        # Windows: use tasklist + taskkill
        try:
            # List grail processes
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python*", "/FO", "CSV"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # This is approximate on Windows - just kill python processes with grail in command line
            # In practice, the bash scripts handle this better
        except Exception:
            pass
    else:
        # Linux/macOS: use pkill
        patterns = ["grail mine", "grail train"]
        for pattern in patterns:
            try:
                # First, find PIDs to exclude self
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                pids = [int(p) for p in result.stdout.strip().split() if p]

                for pid in pids:
                    if exclude_self and pid == my_pid:
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                    except (ProcessLookupError, PermissionError):
                        pass
            except FileNotFoundError:
                # pgrep not available, fall back to pkill
                try:
                    subprocess.run(
                        ["pkill", "-9", "-f", pattern],
                        capture_output=True,
                        timeout=10,
                    )
                    killed += 1
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Error killing {pattern}: {e}")

    if killed > 0:
        logger.info(f"Killed {killed} zombie grail processes")
        time.sleep(grace_period)

    return killed


def cleanup_zombie_processes(
    kill_vllm: bool = True,
    kill_grail: bool = False,
    grace_period: float = 2.0,
) -> int:
    """Clean up zombie processes before starting miners.

    This is called automatically at miner startup to ensure GPU memory
    is available and no stale processes interfere.

    Args:
        kill_vllm: If True, kill zombie vLLM processes
        kill_grail: If True, kill zombie grail processes (use carefully!)
        grace_period: Seconds to wait for GPU memory release

    Returns:
        Total number of process groups killed
    """
    total_killed = 0

    if kill_vllm:
        total_killed += kill_zombie_vllm_processes(grace_period=0)

    if kill_grail:
        total_killed += kill_zombie_grail_processes(grace_period=0)

    if total_killed > 0:
        logger.info(
            f"Cleanup complete: killed {total_killed} zombie process groups, "
            f"waiting {grace_period}s for GPU memory release..."
        )
        time.sleep(grace_period)

    return total_killed


def cleanup_vllm_for_port(port: int, grace_period: float = 2.0) -> bool:
    """Kill any vLLM process using a specific port.

    This is used before starting a new vLLM server to ensure the port
    is available and any zombie server is cleaned up.

    Args:
        port: Port number to check
        grace_period: Seconds to wait after killing

    Returns:
        True if a process was killed
    """
    if _is_windows():
        # Windows: use netstat + taskkill
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                timeout=10,
                            )
                            logger.info(f"Killed process {pid} using port {port}")
                            time.sleep(grace_period)
                            return True
                        except Exception:
                            pass
        except Exception:
            pass
    else:
        # Linux: use lsof or fuser
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            pids = [int(p) for p in result.stdout.strip().split() if p]
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Killed process {pid} using port {port}")
                except (ProcessLookupError, PermissionError):
                    pass
            if pids:
                time.sleep(grace_period)
                return True
        except FileNotFoundError:
            # lsof not available, try fuser
            try:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True,
                    timeout=10,
                )
                time.sleep(grace_period)
                return True
            except Exception:
                pass
        except Exception:
            pass

    return False
