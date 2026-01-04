#!/usr/bin/env python3
"""
Health monitor for GRAIL multi-node mining.

Monitors miner logs for issues and can trigger restarts:
- Insufficient rollouts (< 5120 per window)
- Gaps in problem indices (causes truncation)
- Upload failures
- Checkpoint download failures / excessive retries
- Skipped mining due to stale checkpoint (would cause SKETCH MISMATCH)

Usage:
    # Use grailv alias to enter venv and project root, then:
    python scripts/health_monitor.py

    # Monitor and auto-restart local PM2 on issues
    python scripts/health_monitor.py --auto-restart

    # Monitor with custom thresholds
    python scripts/health_monitor.py --min-rollouts 4096

Environment variables:
    GRAIL_HEALTH_MIN_ROLLOUTS - Minimum rollouts per window (default: 5120)
    GRAIL_HEALTH_LOG_DIR - Log directory (default: /var/log/grail)
    GRAIL_HEALTH_CHECK_INTERVAL - Check interval in seconds (default: 30)
    GRAIL_HEALTH_AUTO_RESTART - Enable auto-restart (default: false)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default thresholds
DEFAULT_MIN_ROLLOUTS = int(os.getenv("GRAIL_HEALTH_MIN_ROLLOUTS", "5120"))
DEFAULT_LOG_DIR = os.getenv("GRAIL_HEALTH_LOG_DIR", "/var/log/grail")
DEFAULT_CHECK_INTERVAL = int(os.getenv("GRAIL_HEALTH_CHECK_INTERVAL", "30"))
AUTO_RESTART = os.getenv("GRAIL_HEALTH_AUTO_RESTART", "").lower() in ("1", "true", "yes")

# Patterns to match in logs
PATTERNS = {
    # [SUMMARY] W0 | window=7237200 | rollouts=5432 | UPLOADED | 12:34:56
    "summary": re.compile(
        r"\[SUMMARY\]\s+(\S+)\s+\|\s+window=(\d+)\s+\|\s+rollouts=(\d+)\s+\|\s+(\w+)\s+\|"
    ),
    # Hub aggregated 41234 rollouts from all workers
    "hub_aggregated": re.compile(
        r"Hub aggregated (\d+) (?:total )?rollouts from (?:all workers|\d+ workers)"
    ),
    # Hub: gap at problem 123, truncated from 5000 to 4000 contiguous rollouts (lost 1000)
    "gap_detected": re.compile(
        r"Hub: gap at problem (\d+), truncated from (\d+) to (\d+) contiguous rollouts \(lost (\d+)\)"
    ),
    # Hub: removed 50 duplicate rollouts from aggregation
    "duplicates": re.compile(
        r"Hub: removed (\d+) duplicate rollouts"
    ),
    # Failed to upload window
    "upload_failed": re.compile(
        r"Failed to upload window (\d+)"
    ),
    # Checkpoint download failed
    "checkpoint_failed": re.compile(
        r"Checkpoint download failed for window (\d+)"
    ),
    # Checkpoint file not found (retrying) - precursor to potential issues
    "checkpoint_retry": re.compile(
        r"File .+\.safetensors not found \(attempt (\d+)/(\d+)\)"
    ),
    # Skipping mining due to checkpoint issues (would cause sketch mismatch)
    "skipping_mining": re.compile(
        r"Skipping mining.*checkpoint|CRITICAL.*stale model"
    ),
}


@dataclass
class WindowHealth:
    """Health status for a single window."""
    window: int
    rollout_count: int = 0
    status: str = "unknown"
    gaps_detected: bool = False
    gap_problem_index: Optional[int] = None
    rollouts_lost_to_gap: int = 0
    duplicates_removed: int = 0
    upload_failed: bool = False
    checkpoint_failed: bool = False
    checkpoint_retries: int = 0  # Count of retry attempts
    skipped_mining: bool = False  # Skipped due to stale checkpoint
    timestamp: str = ""
    worker_summaries: dict = field(default_factory=dict)

    def is_healthy(self, min_rollouts: int) -> bool:
        """Check if this window is healthy."""
        if self.rollout_count < min_rollouts:
            return False
        if self.gaps_detected:
            return False
        if self.upload_failed:
            return False
        if self.checkpoint_failed:
            return False
        if self.skipped_mining:
            return False
        return True

    def get_issues(self, min_rollouts: int) -> list[str]:
        """Get list of issues for this window."""
        issues = []
        if self.rollout_count < min_rollouts:
            issues.append(f"LOW_ROLLOUTS: {self.rollout_count} < {min_rollouts}")
        if self.gaps_detected:
            issues.append(f"GAP_AT_PROBLEM_{self.gap_problem_index}: lost {self.rollouts_lost_to_gap} rollouts")
        if self.duplicates_removed > 0:
            issues.append(f"DUPLICATES_REMOVED: {self.duplicates_removed}")
        if self.upload_failed:
            issues.append("UPLOAD_FAILED")
        if self.checkpoint_failed:
            issues.append("CHECKPOINT_DOWNLOAD_FAILED")
        if self.checkpoint_retries > 0:
            issues.append(f"CHECKPOINT_RETRIES: {self.checkpoint_retries}")
        if self.skipped_mining:
            issues.append("SKIPPED_MINING_STALE_CHECKPOINT")
        return issues


class HealthMonitor:
    """Monitor miner health from log files."""

    def __init__(
        self,
        log_dir: str,
        min_rollouts: int = DEFAULT_MIN_ROLLOUTS,
        auto_restart: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.min_rollouts = min_rollouts
        self.auto_restart = auto_restart

        # Track window health
        self.windows: dict[int, WindowHealth] = {}

        # Track file positions for tailing
        self.file_positions: dict[Path, int] = {}

        # Track consecutive failures for restart decision
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

        # Track last restart time to avoid restart loops
        self.last_restart_time: Optional[float] = None
        self.min_restart_interval = 300  # 5 minutes minimum between restarts

    def get_log_files(self) -> list[Path]:
        """Get all log files to monitor."""
        if not self.log_dir.exists():
            print(f"Warning: Log directory {self.log_dir} does not exist")
            return []

        # Monitor both out and error logs for worker 0 (hub)
        files = []
        for pattern in ["worker-0-out.log", "worker-0-error.log"]:
            path = self.log_dir / pattern
            if path.exists():
                files.append(path)

        return files

    def read_new_lines(self, log_file: Path) -> list[str]:
        """Read new lines from a log file since last check."""
        if not log_file.exists():
            return []

        current_size = log_file.stat().st_size
        last_pos = self.file_positions.get(log_file, 0)

        # Handle log rotation (file got smaller)
        if current_size < last_pos:
            last_pos = 0

        if current_size == last_pos:
            return []

        lines = []
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                lines = f.readlines()
                self.file_positions[log_file] = f.tell()
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

        return lines

    def parse_line(self, line: str) -> None:
        """Parse a log line and update health status."""
        # Check for summary line
        match = PATTERNS["summary"].search(line)
        if match:
            worker_key, window_str, rollouts_str, status = match.groups()
            window = int(window_str)
            rollouts = int(rollouts_str)

            if window not in self.windows:
                self.windows[window] = WindowHealth(window=window)

            wh = self.windows[window]
            wh.worker_summaries[worker_key] = rollouts

            # Update total for hub (W0) uploads
            if status == "UPLOADED" and worker_key == "W0":
                wh.rollout_count = rollouts
                wh.status = status
                wh.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return

        # Check for hub aggregated - PRIMARY source of rollout count
        # This fires before upload, so we catch the count even if upload fails
        match = PATTERNS["hub_aggregated"].search(line)
        if match:
            rollouts = int(match.group(1))
            # Find the most recent window
            if self.windows:
                latest_window = max(self.windows.keys())
                wh = self.windows[latest_window]
                wh.rollout_count = rollouts
                # Mark as aggregated so we can check it even if upload fails
                if wh.status == "unknown":
                    wh.status = "aggregated"
            return

        # Check for gap detected
        match = PATTERNS["gap_detected"].search(line)
        if match:
            gap_idx, original, truncated, lost = match.groups()
            if self.windows:
                latest_window = max(self.windows.keys())
                wh = self.windows[latest_window]
                wh.gaps_detected = True
                wh.gap_problem_index = int(gap_idx)
                wh.rollouts_lost_to_gap = int(lost)
            return

        # Check for duplicates
        match = PATTERNS["duplicates"].search(line)
        if match:
            duplicates = int(match.group(1))
            if self.windows:
                latest_window = max(self.windows.keys())
                self.windows[latest_window].duplicates_removed = duplicates
            return

        # Check for upload failure
        match = PATTERNS["upload_failed"].search(line)
        if match:
            window = int(match.group(1))
            if window in self.windows:
                self.windows[window].upload_failed = True
            return

        # Check for checkpoint failure
        match = PATTERNS["checkpoint_failed"].search(line)
        if match:
            window = int(match.group(1))
            if window not in self.windows:
                self.windows[window] = WindowHealth(window=window)
            self.windows[window].checkpoint_failed = True
            return

        # Check for checkpoint retry warnings
        match = PATTERNS["checkpoint_retry"].search(line)
        if match:
            if self.windows:
                latest_window = max(self.windows.keys())
                self.windows[latest_window].checkpoint_retries += 1
            return

        # Check for skipped mining due to stale checkpoint
        if PATTERNS["skipping_mining"].search(line):
            if self.windows:
                latest_window = max(self.windows.keys())
                self.windows[latest_window].skipped_mining = True
            return

    def check_health(self) -> list[WindowHealth]:
        """Check health of recent windows and return unhealthy ones."""
        # Read new lines from all log files
        for log_file in self.get_log_files():
            for line in self.read_new_lines(log_file):
                self.parse_line(line)

        # Check recent windows (last 5)
        unhealthy = []
        recent_windows = sorted(self.windows.keys())[-5:]

        for window in recent_windows:
            wh = self.windows[window]
            # Check windows that are at least aggregated (or uploaded)
            # This catches issues even if upload fails after aggregation
            if wh.status in ("aggregated", "UPLOADED") and not wh.is_healthy(self.min_rollouts):
                unhealthy.append(wh)

        return unhealthy

    def restart_pm2(self) -> bool:
        """Restart local PM2 grail processes using start-miners.sh."""
        # Check restart cooldown
        if self.last_restart_time:
            elapsed = time.time() - self.last_restart_time
            if elapsed < self.min_restart_interval:
                print(f"Restart cooldown: {self.min_restart_interval - elapsed:.0f}s remaining")
                return False

        # Auto-detect GRAIL_HOME from script location (scripts/ is inside GRAIL_HOME)
        script_dir = Path(__file__).resolve().parent
        grail_home = os.getenv("GRAIL_HOME", str(script_dir.parent))

        print(f"Restarting miners via start-miners.sh (GRAIL_HOME={grail_home})...")
        try:
            result = subprocess.run(
                ["bash", "scripts/start-miners.sh", "current.config.js"],
                capture_output=True,
                text=True,
                timeout=120,  # Script handles stop + start
                cwd=grail_home,
            )

            if result.returncode == 0:
                print("Restart successful")
                self.last_restart_time = time.time()
                self.consecutive_failures = 0
                return True
            else:
                print(f"Restart failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("Restart timed out")
            return False
        except FileNotFoundError:
            print("bash or start-miners.sh not found")
            return False
        except Exception as e:
            print(f"Restart error: {e}")
            return False

    def print_status(self, unhealthy: list[WindowHealth]) -> None:
        """Print health status."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not unhealthy:
            # Get last healthy window
            if self.windows:
                latest = max(self.windows.keys())
                wh = self.windows[latest]
                if wh.status == "UPLOADED":
                    print(f"[{ts}] OK - Window {latest}: {wh.rollout_count} rollouts")
                else:
                    print(f"[{ts}] PENDING - Window {latest} in progress")
            else:
                print(f"[{ts}] WAITING - No windows processed yet")
            return

        # Print unhealthy windows
        for wh in unhealthy:
            issues = wh.get_issues(self.min_rollouts)
            issues_str = ", ".join(issues)
            print(f"[{ts}] UNHEALTHY - Window {wh.window}: {issues_str}")

    def run(self, check_interval: int = DEFAULT_CHECK_INTERVAL) -> None:
        """Run the health monitor continuously."""
        print(f"Starting health monitor...")
        print(f"  Log directory: {self.log_dir}")
        print(f"  Min rollouts: {self.min_rollouts}")
        print(f"  Auto-restart: {self.auto_restart}")
        print(f"  Check interval: {check_interval}s")
        print()

        while True:
            try:
                unhealthy = self.check_health()
                self.print_status(unhealthy)

                if unhealthy:
                    self.consecutive_failures += 1

                    if self.auto_restart and self.consecutive_failures >= self.max_consecutive_failures:
                        print(f"Triggering restart after {self.consecutive_failures} consecutive failures")
                        self.restart_pm2()
                else:
                    self.consecutive_failures = 0

                time.sleep(check_interval)

            except KeyboardInterrupt:
                print("\nStopping health monitor")
                break
            except Exception as e:
                print(f"Error in health check: {e}")
                time.sleep(check_interval)

    def check_once(self) -> int:
        """Run a single health check and return exit code."""
        unhealthy = self.check_health()
        self.print_status(unhealthy)

        if unhealthy:
            return 1
        return 0


def main():
    parser = argparse.ArgumentParser(description="GRAIL Miner Health Monitor")
    parser.add_argument(
        "--log-dir", "-l",
        default=DEFAULT_LOG_DIR,
        help=f"Log directory to monitor (default: {DEFAULT_LOG_DIR})"
    )
    parser.add_argument(
        "--min-rollouts", "-m",
        type=int,
        default=DEFAULT_MIN_ROLLOUTS,
        help=f"Minimum rollouts per window (default: {DEFAULT_MIN_ROLLOUTS})"
    )
    parser.add_argument(
        "--auto-restart", "-r",
        action="store_true",
        default=AUTO_RESTART,
        help="Auto-restart PM2 on repeated failures"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=DEFAULT_CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {DEFAULT_CHECK_INTERVAL})"
    )
    parser.add_argument(
        "--once", "-1",
        action="store_true",
        help="Run single check and exit (for cron/systemd)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    monitor = HealthMonitor(
        log_dir=args.log_dir,
        min_rollouts=args.min_rollouts,
        auto_restart=args.auto_restart,
    )

    if args.once:
        sys.exit(monitor.check_once())
    else:
        monitor.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
