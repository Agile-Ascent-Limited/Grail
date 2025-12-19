#!/usr/bin/env python3
"""Quick validation script to test multi-GPU and optimization setup.

This script validates that your configuration is working correctly before
running the full benchmark or connecting to the network.

Usage:
    # Basic validation
    python scripts/test_multi_gpu_setup.py

    # Test with specific model
    python scripts/test_multi_gpu_setup.py --model Qwen/Qwen2.5-3B-Instruct

    # Test multi-GPU configuration
    GRAIL_MULTI_GPU=1 python scripts/test_multi_gpu_setup.py

    # Test with vLLM (requires vLLM server running)
    GRAIL_USE_VLLM=1 python scripts/test_multi_gpu_setup.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add grail to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {text}")
    print("=" * 60)


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def test_cuda():
    """Test CUDA availability."""
    print_header("CUDA Configuration")

    import torch

    cuda_available = torch.cuda.is_available()
    print_check("CUDA available", cuda_available)

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print_check("Multiple GPUs", num_gpus > 1, f"Found {num_gpus} GPU(s)")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(
                f"         GPU {i}: {props.name} | "
                f"{free_mem / 1024**3:.1f}/{total_mem / 1024**3:.1f} GB free | "
                f"Compute: {props.major}.{props.minor}"
            )

    return cuda_available


def test_flash_attention():
    """Test Flash Attention availability."""
    print_header("Flash Attention")

    try:
        import flash_attn

        version = getattr(flash_attn, "__version__", "unknown")
        print_check("flash-attn installed", True, f"Version: {version}")
        return True
    except ImportError:
        print_check(
            "flash-attn installed",
            False,
            "Install with: pip install flash-attn --no-build-isolation",
        )
        return False


def test_environment_config():
    """Test environment configuration."""
    print_header("Environment Configuration")

    config = {
        "GRAIL_MULTI_GPU": os.getenv("GRAIL_MULTI_GPU", "0"),
        "GRAIL_TENSOR_PARALLEL_SIZE": os.getenv("GRAIL_TENSOR_PARALLEL_SIZE", "0"),
        "GRAIL_USE_FLASH_ATTENTION": os.getenv("GRAIL_USE_FLASH_ATTENTION", "0"),
        "GRAIL_GENERATION_BATCH_SIZE": os.getenv("GRAIL_GENERATION_BATCH_SIZE", "2"),
        "GRAIL_USE_VLLM": os.getenv("GRAIL_USE_VLLM", "0"),
        "GRAIL_VLLM_URL": os.getenv("GRAIL_VLLM_URL", "http://localhost:8000"),
        "GRAIL_WORKER_ID": os.getenv("GRAIL_WORKER_ID", "0"),
        "GRAIL_TOTAL_WORKERS": os.getenv("GRAIL_TOTAL_WORKERS", "1"),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
    }

    for key, value in config.items():
        print(f"  {key}: {value}")

    return config


def test_model_loading(model_name: str):
    """Test model loading with current configuration."""
    print_header(f"Model Loading: {model_name}")

    try:
        from grail.model.provider import get_model, get_tokenizer

        import torch

        # Track memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

        print("  Loading model...")
        model = get_model(model_name, device=None, eval_mode=True)
        tokenizer = get_tokenizer(model_name)

        # Check model attributes
        has_primary_device = hasattr(model, "grail_primary_device")
        primary_device = getattr(model, "grail_primary_device", "unknown")

        print_check("Model loaded", True)
        print_check("Primary device set", has_primary_device, f"Device: {primary_device}")

        # Check memory usage
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"         Memory used: {(mem_after - mem_before) / 1024**3:.2f} GB")
            print(f"         Peak memory: {peak_mem / 1024**3:.2f} GB")

        # Check if model is distributed
        try:
            device_map = getattr(model, "hf_device_map", None)
            if device_map:
                unique_devices = set(device_map.values())
                print_check(
                    "Model distributed",
                    len(unique_devices) > 1,
                    f"Across {len(unique_devices)} device(s)",
                )
        except Exception:
            pass

        return model, tokenizer

    except Exception as e:
        print_check("Model loaded", False, str(e))
        return None, None


def test_single_generation(model, tokenizer):
    """Test a single rollout generation."""
    print_header("Single Rollout Generation")

    if model is None or tokenizer is None:
        print_check("Generation test", False, "Model not loaded")
        return False

    try:
        from grail.environments.factory import create_env
        from grail.environments.loop import AgentEnvLoop
        from grail.shared.constants import ROLLOUTS_PER_PROBLEM

        import time

        device = getattr(model, "grail_primary_device", "cuda")
        loop = AgentEnvLoop(model, tokenizer, device)

        def _env_factory():
            return create_env()

        print("  Generating rollouts (this may take a moment)...")
        start = time.time()

        rollouts = loop.run_grpo_group(
            _env_factory,
            ROLLOUTS_PER_PROBLEM,
            "test_randomness_12345",
            wallet=None,
            batch_size=4,
            seed=42,
        )

        elapsed = time.time() - start

        print_check("Generation completed", True, f"Generated {len(rollouts)} rollouts")
        print(f"         Time: {elapsed:.2f}s ({len(rollouts) / elapsed:.1f} rollouts/s)")

        # Check rollout quality
        successful = sum(1 for r in rollouts if r.success)
        print(f"         Successful: {successful}/{len(rollouts)}")

        # Check first rollout has expected fields
        if rollouts:
            r = rollouts[0]
            has_tokens = hasattr(r, "tokens") and len(r.tokens) > 0
            has_commitments = hasattr(r, "commitments") and len(r.commitments) > 0
            has_reward = hasattr(r, "reward")

            print_check("Tokens generated", has_tokens, f"{len(r.tokens) if has_tokens else 0} tokens")
            print_check(
                "Commitments generated",
                has_commitments,
                f"{len(r.commitments) if has_commitments else 0} commitments",
            )
            print_check("Reward computed", has_reward, f"Reward: {r.reward:.3f}" if has_reward else "")

        return True

    except Exception as e:
        print_check("Generation test", False, str(e))
        import traceback

        traceback.print_exc()
        return False


def test_worker_config():
    """Test worker configuration."""
    print_header("Worker Configuration")

    try:
        from grail.infrastructure.worker_config import WorkerConfig

        config = WorkerConfig.from_env()

        print_check("Worker config loaded", True)
        print(f"         Worker ID: {config.worker_id}")
        print(f"         Total workers: {config.total_workers}")

        # Test problem partitioning
        if config.total_workers > 1:
            my_problems = [i for i in range(16) if config.should_handle_problem(i)]
            print(f"         This worker handles problems: {my_problems}")

        return True

    except Exception as e:
        print_check("Worker config", False, str(e))
        return False


def test_vllm_backend():
    """Test vLLM backend if enabled."""
    if os.getenv("GRAIL_USE_VLLM", "0") != "1":
        return True

    print_header("vLLM Backend")

    try:
        import httpx

        url = os.getenv("GRAIL_VLLM_URL", "http://localhost:8000")
        response = httpx.get(f"{url}/health", timeout=5.0)

        print_check("vLLM server reachable", response.status_code == 200, f"URL: {url}")
        return response.status_code == 200

    except Exception as e:
        print_check("vLLM server reachable", False, str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="Test GRAIL multi-GPU setup")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to test",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip the generation test (faster)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" GRAIL MULTI-GPU SETUP VALIDATION")
    print("=" * 60)

    results = {}

    # Run tests
    results["cuda"] = test_cuda()
    results["flash_attention"] = test_flash_attention()
    results["config"] = test_environment_config()
    results["worker"] = test_worker_config()
    results["vllm"] = test_vllm_backend()

    model, tokenizer = test_model_loading(args.model)
    results["model"] = model is not None

    if not args.skip_generation and model is not None:
        results["generation"] = test_single_generation(model, tokenizer)
    else:
        results["generation"] = None

    # Summary
    print_header("SUMMARY")

    all_passed = all(v for v in results.values() if v is not None)
    critical_passed = results["cuda"] and results["model"]

    if all_passed:
        print("  ✅ All tests passed! Your setup is ready for mining.")
    elif critical_passed:
        print("  ⚠️  Core tests passed but some optional features failed.")
        print("     You can still mine, but may not have optimal performance.")
    else:
        print("  ❌ Critical tests failed. Please fix issues before mining.")

    print("\nRecommended next steps:")
    if results["cuda"] and not results.get("flash_attention"):
        print("  - Install flash-attn for ~30% faster generation")
    if os.getenv("GRAIL_MULTI_GPU", "0") == "0":
        print("  - Consider GRAIL_MULTI_GPU=1 for multi-GPU support")
    if int(os.getenv("GRAIL_GENERATION_BATCH_SIZE", "2")) < 8:
        print("  - Try GRAIL_GENERATION_BATCH_SIZE=8 for better throughput")

    print("\nRun the full benchmark with:")
    print("  python scripts/benchmark_throughput.py --full-benchmark")


if __name__ == "__main__":
    main()
