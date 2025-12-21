"""Multi-worker configuration for parallel mining.

Enables running multiple miner processes on the same machine, each handling
a subset of problems. This allows linear scaling with GPU count.

Usage:
    # Worker 0 of 4 workers (handles problems 0, 4, 8, 12, ...)
    GRAIL_WORKER_ID=0 GRAIL_TOTAL_WORKERS=4 CUDA_VISIBLE_DEVICES=0 grail mine

    # Worker 1 of 4 workers (handles problems 1, 5, 9, 13, ...)
    GRAIL_WORKER_ID=1 GRAIL_TOTAL_WORKERS=4 CUDA_VISIBLE_DEVICES=1 grail mine

    # And so on for workers 2, 3...

For 8x A100:
    - Option A: 8 workers (1 GPU each) - best for 4B model
    - Option B: 4 workers (2 GPUs each with tensor parallel) - better for 30B model
    - Option C: 2 workers (4 GPUs each) - for very large models
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for a single worker in a multi-worker setup."""

    worker_id: int
    total_workers: int

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Load worker configuration from environment variables.

        Environment Variables:
            GRAIL_WORKER_ID: This worker's ID (0-indexed, default: 0)
            GRAIL_TOTAL_WORKERS: Total number of workers (default: 1)

        Returns:
            WorkerConfig instance
        """
        worker_id_str = os.getenv("GRAIL_WORKER_ID", "0")
        total_workers_str = os.getenv("GRAIL_TOTAL_WORKERS", "1")
        worker_id = int(worker_id_str)
        total_workers = int(total_workers_str)

        # Debug logging for environment variable issues
        logger.info(
            "WorkerConfig: GRAIL_WORKER_ID=%r, GRAIL_TOTAL_WORKERS=%r (parsed: %d/%d)",
            worker_id_str,
            total_workers_str,
            worker_id,
            total_workers,
        )

        if worker_id < 0 or worker_id >= total_workers:
            raise ValueError(
                f"Invalid worker config: worker_id={worker_id}, total_workers={total_workers}. "
                f"worker_id must be in range [0, {total_workers - 1}]"
            )

        config = cls(worker_id=worker_id, total_workers=total_workers)

        if total_workers > 1:
            logger.info(
                f"🔧 Multi-worker mode: Worker {worker_id + 1}/{total_workers} "
                f"(handling problems {worker_id}, {worker_id + total_workers}, ...)"
            )

        return config

    def should_handle_problem(self, problem_index: int) -> bool:
        """Check if this worker should handle a given problem index.

        Uses round-robin distribution: worker i handles problems where
        problem_index % total_workers == worker_id

        Args:
            problem_index: The problem index (0-indexed)

        Returns:
            True if this worker should process the problem
        """
        if self.total_workers == 1:
            return True
        return (problem_index % self.total_workers) == self.worker_id

    def get_worker_problem_index(self, global_problem_index: int) -> int:
        """Convert global problem index to worker-local index.

        Args:
            global_problem_index: The global problem count

        Returns:
            The worker's local problem count
        """
        if self.total_workers == 1:
            return global_problem_index
        # Count how many problems this worker has handled
        return global_problem_index // self.total_workers

    def is_single_worker(self) -> bool:
        """Check if running in single-worker mode."""
        return self.total_workers == 1


def get_worker_config() -> WorkerConfig:
    """Get the current worker configuration.

    Returns:
        WorkerConfig instance from environment
    """
    return WorkerConfig.from_env()


def log_multi_gpu_setup() -> None:
    """Log information about the multi-GPU setup."""
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - running on CPU")
        return

    num_gpus = torch.cuda.device_count()
    logger.info(f"🖥️  Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free_mem = torch.cuda.mem_get_info(i)[0] / 1024**3
        total_mem = props.total_memory / 1024**3
        logger.info(
            f"  GPU {i}: {props.name} | "
            f"{free_mem:.1f}/{total_mem:.1f} GB free | "
            f"Compute capability: {props.major}.{props.minor}"
        )

    # Check environment setup
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "all")
    logger.info(f"  CUDA_VISIBLE_DEVICES: {visible_devices}")

    # Log worker config
    worker_id = os.getenv("GRAIL_WORKER_ID", "0")
    total_workers = os.getenv("GRAIL_TOTAL_WORKERS", "1")
    logger.info(f"  Worker: {worker_id}/{total_workers}")

    # Log multi-GPU and vLLM settings
    multi_gpu = os.getenv("GRAIL_MULTI_GPU", "0")
    use_vllm = os.getenv("GRAIL_USE_VLLM", "0")
    flash_attn = os.getenv("GRAIL_USE_FLASH_ATTENTION", "0")

    logger.info(
        f"  Settings: MULTI_GPU={multi_gpu}, USE_VLLM={use_vllm}, FLASH_ATTENTION={flash_attn}"
    )
