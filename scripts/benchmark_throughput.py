#!/usr/bin/env python3
"""Throughput benchmark for GRAIL mining.

This script measures rollout generation throughput to help tune your setup.
Requires a configured bittensor wallet (via BT_WALLET_COLD/BT_WALLET_HOT env vars).

Usage:
    # Basic benchmark (uses defaults)
    python scripts/benchmark_throughput.py

    # Test with specific model
    python scripts/benchmark_throughput.py --model Qwen/Qwen2.5-3B-Instruct

    # Test multi-GPU
    GRAIL_MULTI_GPU=1 python scripts/benchmark_throughput.py

    # Test specific batch sizes
    python scripts/benchmark_throughput.py --batch-sizes 1,2,4,8,16

    # Full benchmark suite
    python scripts/benchmark_throughput.py --full-benchmark

    # Simulate multi-worker (measure what each worker would produce)
    python scripts/benchmark_throughput.py --workers 8
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add grail to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import bittensor as bt
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    batch_size: int
    num_problems: int
    total_rollouts: int
    total_time_s: float
    rollouts_per_second: float
    problems_per_second: float
    avg_time_per_problem_s: float
    gpu_memory_gb: float
    successful_rollouts: int
    success_rate: float


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "devices": [],
    }

    for i in range(info["count"]):
        props = torch.cuda.get_device_properties(i)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        info["devices"].append(
            {
                "id": i,
                "name": props.name,
                "total_memory_gb": total_mem / 1024**3,
                "free_memory_gb": free_mem / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )

    return info


def load_wallet() -> bt.wallet:
    """Load wallet from environment variables."""
    coldkey = os.getenv("BT_WALLET_COLD", "default")
    hotkey = os.getenv("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)
    logger.info(f"Loaded wallet: {coldkey}/{hotkey} ({wallet.hotkey.ss58_address})")
    return wallet


def load_model_for_benchmark(model_name: str):
    """Load model with current environment configuration."""
    from grail.model.provider import get_model, get_tokenizer

    logger.info(f"Loading model: {model_name}")
    start = time.time()

    model = get_model(model_name, device=None, eval_mode=True)
    tokenizer = get_tokenizer(model_name)

    load_time = time.time() - start
    logger.info(f"Model loaded in {load_time:.1f}s")

    return model, tokenizer


def run_benchmark(
    model,
    tokenizer,
    wallet: bt.wallet,
    batch_size: int,
    num_problems: int,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a single benchmark with given configuration."""
    from grail.environments.factory import create_env
    from grail.environments.loop import AgentEnvLoop
    from grail.shared.constants import ROLLOUTS_PER_PROBLEM

    # Detect device
    device = getattr(model, "grail_primary_device", None) or "cuda"

    logger.info(f"Running benchmark: batch_size={batch_size}, problems={num_problems}")

    # Create the loop
    loop = AgentEnvLoop(model, tokenizer, device)

    # Warm up (1 problem)
    logger.info("Warming up...")
    try:

        def _env_factory():
            return create_env()

        # Generate valid hex randomness for warmup
        warmup_hex = hashlib.sha256(b"warmup_randomness_12345").hexdigest()
        _ = loop.run_grpo_group(
            _env_factory,
            ROLLOUTS_PER_PROBLEM,
            warmup_hex,
            wallet=wallet,
            batch_size=min(batch_size, ROLLOUTS_PER_PROBLEM),
            seed=seed,
        )
    except Exception as e:
        logger.warning(f"Warmup failed (this is okay for benchmark): {e}")

    # Clear cache before benchmark
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Run benchmark
    total_rollouts = 0
    successful_rollouts = 0
    start_time = time.time()

    for problem_idx in range(num_problems):
        try:

            def _env_factory():
                return create_env()

            # Generate valid hex randomness for this problem
            problem_hex = hashlib.sha256(f"benchmark_randomness_{problem_idx}".encode()).hexdigest()
            rollouts = loop.run_grpo_group(
                _env_factory,
                ROLLOUTS_PER_PROBLEM,
                problem_hex,
                wallet=wallet,
                batch_size=min(batch_size, ROLLOUTS_PER_PROBLEM),
                seed=seed + problem_idx,
            )

            total_rollouts += len(rollouts)
            successful_rollouts += sum(1 for r in rollouts if r.success)

            if (problem_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                rate = total_rollouts / elapsed
                logger.info(
                    f"  Progress: {problem_idx + 1}/{num_problems} problems, "
                    f"{total_rollouts} rollouts, {rate:.1f} rollouts/s"
                )

        except Exception as e:
            logger.error(f"Problem {problem_idx} failed: {e}")
            continue

    total_time = time.time() - start_time

    # Get GPU memory usage
    gpu_memory_gb = 0.0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return BenchmarkResult(
        batch_size=batch_size,
        num_problems=num_problems,
        total_rollouts=total_rollouts,
        total_time_s=total_time,
        rollouts_per_second=total_rollouts / total_time if total_time > 0 else 0,
        problems_per_second=num_problems / total_time if total_time > 0 else 0,
        avg_time_per_problem_s=total_time / num_problems if num_problems > 0 else 0,
        gpu_memory_gb=gpu_memory_gb,
        successful_rollouts=successful_rollouts,
        success_rate=successful_rollouts / total_rollouts if total_rollouts > 0 else 0,
    )


def print_results(results: list[BenchmarkResult], workers: int = 1):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print(
        f"\n{'Batch':<8} {'Problems':<10} {'Rollouts':<10} {'Time(s)':<10} "
        f"{'Roll/s':<10} {'Prob/s':<10} {'GPU(GB)':<10} {'Success%':<10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.batch_size:<8} {r.num_problems:<10} {r.total_rollouts:<10} "
            f"{r.total_time_s:<10.1f} {r.rollouts_per_second:<10.1f} "
            f"{r.problems_per_second:<10.2f} {r.gpu_memory_gb:<10.2f} "
            f"{r.success_rate * 100:<10.1f}"
        )

    if workers > 1:
        print("\n" + "-" * 80)
        print(f"PROJECTED {workers}-WORKER THROUGHPUT:")
        print("-" * 80)
        for r in results:
            projected_rollouts_per_s = r.rollouts_per_second * workers
            print(
                f"  Batch {r.batch_size}: {projected_rollouts_per_s:.1f} rollouts/s "
                f"(~{projected_rollouts_per_s * 180:.0f} rollouts per 30-block window)"
            )

    # Best result
    best = max(results, key=lambda x: x.rollouts_per_second)
    print("\n" + "=" * 80)
    print(f"BEST: batch_size={best.batch_size} -> {best.rollouts_per_second:.1f} rollouts/s")
    if workers > 1:
        print(
            f"       With {workers} workers: ~{best.rollouts_per_second * workers:.1f} rollouts/s"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="GRAIL Mining Throughput Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to benchmark (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="4",
        help="Comma-separated batch sizes to test (e.g., '1,2,4,8')",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=10,
        help="Number of problems per benchmark run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Project throughput for N workers (doesn't actually run workers)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run full benchmark suite (tests all batch sizes 1-16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Print system info
    print("\n" + "=" * 80)
    print("GRAIL MINING THROUGHPUT BENCHMARK")
    print("=" * 80)

    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"\nGPUs: {gpu_info['count']}")
        for dev in gpu_info["devices"]:
            print(
                f"  [{dev['id']}] {dev['name']} - "
                f"{dev['free_memory_gb']:.1f}/{dev['total_memory_gb']:.1f} GB free"
            )
    else:
        print("\nNo GPU available - running on CPU (will be slow)")

    # Print environment config
    print("\nEnvironment Configuration:")
    print(f"  GRAIL_MULTI_GPU: {os.getenv('GRAIL_MULTI_GPU', '0')}")
    print(f"  GRAIL_TENSOR_PARALLEL_SIZE: {os.getenv('GRAIL_TENSOR_PARALLEL_SIZE', '0')}")
    print(f"  GRAIL_USE_FLASH_ATTENTION: {os.getenv('GRAIL_USE_FLASH_ATTENTION', '0')}")
    print(f"  GRAIL_USE_VLLM: {os.getenv('GRAIL_USE_VLLM', '0')}")

    # Parse batch sizes
    if args.full_benchmark:
        batch_sizes = [1, 2, 4, 8, 16]
    else:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print(f"\nBenchmark Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Problems per run: {args.num_problems}")
    print(f"  Projected workers: {args.workers}")

    # Load wallet
    print("\n" + "-" * 80)
    print("Loading wallet...")
    wallet = load_wallet()

    # Load model
    print("\n" + "-" * 80)
    model, tokenizer = load_model_for_benchmark(args.model)

    # Run benchmarks
    results = []
    for batch_size in batch_sizes:
        print("\n" + "-" * 80)
        try:
            result = run_benchmark(
                model,
                tokenizer,
                wallet=wallet,
                batch_size=batch_size,
                num_problems=args.num_problems,
                seed=args.seed,
            )
            results.append(result)

            # Clear memory between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        except Exception as e:
            logger.error(f"Benchmark failed for batch_size={batch_size}: {e}")
            continue

    # Print results
    if results:
        print_results(results, workers=args.workers)
    else:
        print("\nNo successful benchmark runs!")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
