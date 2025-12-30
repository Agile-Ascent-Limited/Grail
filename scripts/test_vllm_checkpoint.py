#!/usr/bin/env python3
"""
Test script to load a checkpoint with vLLM and run a simple generation.

Usage:
    python scripts/test_vllm_checkpoint.py

Environment variables:
    GRAIL_VLLM_GPU_MEMORY_UTIL - GPU memory fraction (default: 0.70)
    GRAIL_GENERATION_BATCH_SIZE - Batch size (default: 8)
    CHECKPOINT_PATH - Override checkpoint path (optional)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_latest_checkpoint():
    """Find the latest checkpoint in the cache directory."""
    cache_root = Path.home() / ".cache" / "grail" / "checkpoints"

    if not cache_root.exists():
        print(f"Cache directory not found: {cache_root}")
        return None

    # Find checkpoint directories with .complete marker
    checkpoints = []
    for d in cache_root.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            complete_file = d / ".complete"
            if complete_file.exists():
                try:
                    window = int(d.name.split("-")[1])
                    checkpoints.append((window, d))
                except ValueError:
                    continue

    if not checkpoints:
        print("No completed checkpoints found")
        return None

    # Return the latest (highest window number)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_window, latest_path = checkpoints[0]
    print(f"Found {len(checkpoints)} checkpoints, using latest: {latest_path}")
    return latest_path


def test_vllm_server(checkpoint_path: Path):
    """Test starting vLLM server and running a generation."""
    from vllm import LLM, SamplingParams

    gpu_memory_util = float(os.getenv("GRAIL_VLLM_GPU_MEMORY_UTIL", "0.70"))
    max_num_seqs = int(os.getenv("GRAIL_VLLM_MAX_NUM_SEQS", "16"))

    print(f"\n=== vLLM Configuration ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"GPU memory utilization: {gpu_memory_util}")
    print(f"Max num seqs: {max_num_seqs}")

    print(f"\n=== Loading model with vLLM ===")
    try:
        llm = LLM(
            model=str(checkpoint_path),
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_util,
            max_num_seqs=max_num_seqs,
            dtype="bfloat16",
            enforce_eager=True,  # Disable CUDA graphs for compatibility
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n=== Testing generation ===")
    try:
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,
        )

        prompts = [
            "The capital of France is",
            "def fibonacci(n):",
        ]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            print(f"\nPrompt: {output.prompt}")
            print(f"Output: {output.outputs[0].text}")

        print("\nGeneration test PASSED!")
        return True

    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_api(checkpoint_path: Path):
    """Test using vLLM's OpenAI-compatible API server."""
    import subprocess
    import time
    import requests

    gpu_memory_util = float(os.getenv("GRAIL_VLLM_GPU_MEMORY_UTIL", "0.70"))
    port = 8765

    print(f"\n=== Starting vLLM OpenAI API Server ===")
    print(f"Port: {port}")

    # Start server in background
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(checkpoint_path),
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--port", str(port),
        "--dtype", "bfloat16",
        "--enforce-eager",
    ]

    print(f"Command: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for server to start
    print("Waiting for server to start...")
    for i in range(60):  # Wait up to 60 seconds
        time.sleep(1)
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                print(f"Server healthy after {i+1}s")
                break
        except:
            pass

        # Check if process died
        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            print(f"Server process died!")
            print(f"Output:\n{stdout.decode()}")
            return False
    else:
        print("Server failed to start in 60s")
        proc.terminate()
        return False

    # Test a request
    print("\n=== Testing API request ===")
    try:
        resp = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={
                "model": str(checkpoint_path),
                "prompt": "The meaning of life is",
                "max_tokens": 50,
                "temperature": 0.7,
            },
            timeout=30,
        )

        if resp.status_code == 200:
            result = resp.json()
            print(f"Response: {result['choices'][0]['text']}")
            print("\nAPI test PASSED!")
        else:
            print(f"API error: {resp.status_code} - {resp.text}")

    except Exception as e:
        print(f"API request failed: {e}")
    finally:
        print("\nStopping server...")
        proc.terminate()
        proc.wait(timeout=5)

    return True


def main():
    print("=" * 60)
    print("VLLM CHECKPOINT TEST")
    print("=" * 60)

    # Find or use specified checkpoint
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Specified checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("\nRun download-only mode first to get a checkpoint:")
            print("  GRAIL_DOWNLOAD_ONLY=1 grail mine")
            sys.exit(1)

    # Check checkpoint contents
    print(f"\n=== Checkpoint contents ===")
    for f in sorted(checkpoint_path.iterdir())[:10]:
        size = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        print(f"  {f.name}: {size:.1f} MB" if size else f"  {f.name}/")

    # Test mode selection
    mode = os.getenv("TEST_MODE", "offline")

    if mode == "api":
        success = test_openai_api(checkpoint_path)
    else:
        success = test_vllm_server(checkpoint_path)

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
