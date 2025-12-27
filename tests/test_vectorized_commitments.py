"""Test that vectorized commitment creation produces identical results to original."""

import time
import torch
import sys
sys.path.insert(0, ".")

from grail.protocol.grail_verifier import (
    GRAILVerifier,
    log_magnitude_bucket,
    log_magnitude_bucket_vectorized,
)
from grail.shared.constants import PROOF_NUM_BUCKETS, PROOF_TOPK


def test_log_magnitude_bucket_equivalence():
    """Test that vectorized bucketing matches scalar version exactly."""
    print("\n=== Testing log_magnitude_bucket equivalence ===")

    # Test cases covering all branches
    test_values = [
        0.0,           # deadzone
        1e-7,          # below deadzone
        1e-6,          # at deadzone boundary
        0.5,           # normal positive
        -0.5,          # normal negative
        1.0,           # log2(2) = 1
        2.0,           # log2(3) ≈ 1.58
        10.0,          # larger value
        -10.0,         # larger negative
        100.0,         # even larger
        float('nan'),  # NaN
        float('inf'),  # positive infinity
        float('-inf'), # negative infinity
    ]

    # Test scalar vs vectorized
    for val in test_values:
        scalar_result = log_magnitude_bucket(val, PROOF_NUM_BUCKETS)

        tensor_val = torch.tensor([val], dtype=torch.float32)
        vectorized_result = log_magnitude_bucket_vectorized(tensor_val, PROOF_NUM_BUCKETS)
        vec_result = vectorized_result[0].item()

        match = scalar_result == vec_result
        status = "✓" if match else "✗"
        print(f"  {status} val={val:12.6g}: scalar={scalar_result:3d}, vectorized={vec_result:3d}")

        if not match:
            raise AssertionError(f"Mismatch for value {val}: scalar={scalar_result}, vectorized={vec_result}")

    print("  All scalar tests passed!")

    # Test batch processing
    batch = torch.tensor(test_values[:-3], dtype=torch.float32)  # Exclude nan/inf for batch
    vectorized_batch = log_magnitude_bucket_vectorized(batch, PROOF_NUM_BUCKETS)

    for i, val in enumerate(test_values[:-3]):
        scalar_result = log_magnitude_bucket(val, PROOF_NUM_BUCKETS)
        vec_result = vectorized_batch[i].item()
        if scalar_result != vec_result:
            raise AssertionError(f"Batch mismatch at index {i}: scalar={scalar_result}, vectorized={vec_result}")

    print("  Batch processing test passed!")


def test_create_commitments_batch_equivalence():
    """Test that batch commitment creation matches per-position version exactly."""
    print("\n=== Testing create_commitments_batch equivalence ===")

    hidden_dim = 2560  # Qwen3-4B hidden size
    seq_len = 512      # Typical sequence length

    # Create verifier
    verifier = GRAILVerifier(hidden_dim=hidden_dim)

    # Generate test hidden states
    torch.manual_seed(42)
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.float32)

    # Generate random r_vec
    r_vec = verifier.generate_r_vec("deadbeef" * 8)

    # Compute using both methods
    print(f"  Computing {seq_len} commitments...")

    # Per-position (original slow method)
    t0 = time.time()
    commitments_loop = []
    for pos in range(seq_len):
        c = verifier.create_commitment(hidden_states[pos], r_vec, pos)
        commitments_loop.append(c)
    loop_time = time.time() - t0
    print(f"  Per-position loop: {loop_time:.3f}s ({seq_len/loop_time:.0f} pos/sec)")

    # Batch (new fast method)
    t0 = time.time()
    commitments_batch = verifier.create_commitments_batch(hidden_states, r_vec)
    batch_time = time.time() - t0
    print(f"  Batched version:   {batch_time:.3f}s ({seq_len/batch_time:.0f} pos/sec)")

    speedup = loop_time / batch_time
    print(f"  Speedup: {speedup:.1f}x")

    # Verify equivalence
    print(f"  Verifying {seq_len} commitments match exactly...")
    mismatches = 0
    for pos in range(seq_len):
        loop_c = commitments_loop[pos]
        batch_c = commitments_batch[pos]

        # Check all fields
        if loop_c["sketch"] != batch_c["sketch"]:
            print(f"    MISMATCH at pos {pos}: sketch {loop_c['sketch']} vs {batch_c['sketch']}")
            mismatches += 1
        if loop_c["indices"] != batch_c["indices"]:
            print(f"    MISMATCH at pos {pos}: indices differ")
            mismatches += 1
        if loop_c["position"] != batch_c["position"]:
            print(f"    MISMATCH at pos {pos}: position {loop_c['position']} vs {batch_c['position']}")
            mismatches += 1

    if mismatches > 0:
        raise AssertionError(f"Found {mismatches} mismatches!")

    print(f"  ✓ All {seq_len} commitments match exactly!")
    return speedup


def test_realistic_mining_scenario():
    """Test with realistic mining parameters (16 rollouts, ~600 tokens each)."""
    print("\n=== Testing realistic mining scenario ===")

    hidden_dim = 2560  # Qwen3-4B
    num_rollouts = 16
    avg_seq_len = 600

    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec("abcd1234" * 8)

    total_loop_time = 0.0
    total_batch_time = 0.0
    total_positions = 0

    for i in range(num_rollouts):
        # Vary sequence lengths slightly
        seq_len = avg_seq_len + (i * 20 - 160)
        hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.float32)
        total_positions += seq_len

        # Loop version
        t0 = time.time()
        for pos in range(seq_len):
            verifier.create_commitment(hidden_states[pos], r_vec, pos)
        total_loop_time += time.time() - t0

        # Batch version
        t0 = time.time()
        verifier.create_commitments_batch(hidden_states, r_vec)
        total_batch_time += time.time() - t0

    print(f"  Total positions processed: {total_positions}")
    print(f"  Loop version:  {total_loop_time:.2f}s ({total_positions/total_loop_time:.0f} pos/sec)")
    print(f"  Batch version: {total_batch_time:.2f}s ({total_positions/total_batch_time:.0f} pos/sec)")
    print(f"  Speedup: {total_loop_time/total_batch_time:.1f}x")

    # Estimate time savings per problem
    time_per_problem_loop = total_loop_time
    time_per_problem_batch = total_batch_time
    savings_per_problem = time_per_problem_loop - time_per_problem_batch
    print(f"\n  Time saved per problem (16 rollouts): {savings_per_problem:.2f}s")
    print(f"  Previous proof time estimate: ~7s")
    print(f"  New proof time estimate: ~{7 - savings_per_problem:.1f}s")


def test_gpu_performance():
    """Test performance on GPU if available."""
    if not torch.cuda.is_available():
        print("\n=== Skipping GPU test (CUDA not available) ===")
        return

    print("\n=== Testing GPU performance ===")

    hidden_dim = 2560
    seq_len = 1024

    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec("gpu_test_" * 8)

    # Move to GPU
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(3):
        verifier.create_commitments_batch(hidden_states, r_vec)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(10):
        verifier.create_commitments_batch(hidden_states, r_vec)
    torch.cuda.synchronize()
    gpu_time = (time.time() - t0) / 10

    print(f"  GPU batch time: {gpu_time*1000:.1f}ms for {seq_len} positions")
    print(f"  Throughput: {seq_len/gpu_time:.0f} positions/sec")


if __name__ == "__main__":
    print("=" * 60)
    print("VECTORIZED COMMITMENT VERIFICATION TESTS")
    print("=" * 60)

    test_log_magnitude_bucket_equivalence()
    speedup = test_create_commitments_batch_equivalence()
    test_realistic_mining_scenario()
    test_gpu_performance()

    print("\n" + "=" * 60)
    print(f"ALL TESTS PASSED! Vectorized version is {speedup:.0f}x faster.")
    print("=" * 60)
