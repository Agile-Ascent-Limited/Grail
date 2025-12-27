#!/usr/bin/env python3
"""Local proof verification test.

Compares miner hidden state extraction against validator behavior to diagnose
sketch mismatches. This script simulates both sides of the proof verification.

Usage:
    python scripts/verify_proof_locally.py --checkpoint /path/to/checkpoint

The script will:
1. Load the model (same as both miner and validator)
2. Generate a sample completion
3. Extract hidden states using BOTH methods:
   - Miner method (from loop.py)
   - Validator method (from proof.py)
4. Compare the hidden states and sketch values
5. Report any differences
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from grail.protocol.grail_verifier import (
    GRAILVerifier,
    log_magnitude_bucket,
    log_magnitude_bucket_vectorized,
)
from grail.shared.constants import LAYER_INDEX, PROOF_TOPK

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load model from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Model loaded: {model.config.name_or_path}")
    logger.info(f"Hidden size: {model.config.hidden_size}")
    logger.info(f"Num layers: {model.config.num_hidden_layers}")

    return model, tokenizer


def extract_hidden_states_validator_style(
    model, token_ids: list[int], device: str = "cuda:0"
) -> torch.Tensor:
    """Extract hidden states exactly as the validator does.

    See grail/validation/validators/proof.py:217
    """
    full_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.inference_mode():
        # CRITICAL: No attention_mask, no position_ids - just like validator
        outs = model(full_ids, output_hidden_states=True)

    # Get layer at LAYER_INDEX (typically -1, which is the last layer)
    h_layer = outs.hidden_states[LAYER_INDEX][0]  # Remove batch dim

    return h_layer


def extract_hidden_states_miner_style(
    model, token_ids: list[int], device: str = "cuda:0"
) -> torch.Tensor:
    """Extract hidden states exactly as the miner does.

    See grail/environments/loop.py:1464-1476
    """
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.inference_mode():
        # CRITICAL: No attention_mask, no position_ids - uses model defaults
        # This ensures hidden states match validator's computation exactly
        model_outputs = model(
            token_tensor,
            output_hidden_states=True,
        )

    # hidden_states shape: [1, seq_len, hidden_dim]
    h_layer = model_outputs.hidden_states[LAYER_INDEX][0]  # Remove batch dim

    return h_layer


def compare_hidden_states(
    miner_hidden: torch.Tensor,
    validator_hidden: torch.Tensor,
    positions_to_check: list[int] | None = None,
) -> dict:
    """Compare hidden states from miner and validator."""

    seq_len = miner_hidden.size(0)
    hidden_dim = miner_hidden.size(1)

    if positions_to_check is None:
        # Check a sample of positions
        positions_to_check = [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]
        positions_to_check = [p for p in positions_to_check if p < seq_len]

    results = {
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "positions_checked": positions_to_check,
        "identical": True,
        "max_diff": 0.0,
        "position_results": [],
    }

    for pos in positions_to_check:
        miner_vec = miner_hidden[pos]
        validator_vec = validator_hidden[pos]

        # Check exact equality
        exact_match = torch.allclose(miner_vec, validator_vec, rtol=0, atol=0)

        # Check approximate equality
        approx_match = torch.allclose(miner_vec, validator_vec, rtol=1e-5, atol=1e-5)

        # Compute differences
        diff = torch.abs(miner_vec - validator_vec)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Compare norms
        miner_norm = miner_vec.norm().item()
        validator_norm = validator_vec.norm().item()

        pos_result = {
            "position": pos,
            "exact_match": exact_match,
            "approx_match": approx_match,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "miner_norm": miner_norm,
            "validator_norm": validator_norm,
        }
        results["position_results"].append(pos_result)

        if not exact_match:
            results["identical"] = False
        results["max_diff"] = max(results["max_diff"], max_diff)

    return results


def compare_sketches(
    miner_hidden: torch.Tensor,
    validator_hidden: torch.Tensor,
    r_vec: torch.Tensor,
    positions_to_check: list[int],
    verifier: GRAILVerifier,
) -> dict:
    """Compare sketch values at specific positions."""

    results = {
        "positions": [],
        "all_pass": True,
    }

    for pos in positions_to_check:
        # Create miner commitment (what miner sends)
        miner_commitment = verifier.create_commitment(miner_hidden[pos], r_vec, pos)

        # Verify with validator's hidden state (what validator computes)
        is_valid, diagnostics = verifier.verify_commitment(
            validator_hidden[pos],
            miner_commitment,
            r_vec,
            miner_hidden.size(0),  # seq_len
        )

        # Also compute validator's own commitment for comparison
        validator_commitment = verifier.create_commitment(validator_hidden[pos], r_vec, pos)

        pos_result = {
            "position": pos,
            "is_valid": is_valid,
            "miner_sketch": miner_commitment["sketch"],
            "validator_sketch": validator_commitment["sketch"],
            "sketch_diff": diagnostics["sketch_diff"],
            "tolerance": diagnostics["sketch_tolerance"],
            "miner_indices": miner_commitment["indices"][:5],  # First 5
            "validator_indices": validator_commitment["indices"][:5],
            "indices_match": miner_commitment["indices"] == validator_commitment["indices"],
        }
        results["positions"].append(pos_result)

        if not is_valid:
            results["all_pass"] = False

    return results


def compare_bucketing(miner_hidden: torch.Tensor, validator_hidden: torch.Tensor, pos: int):
    """Compare bucketing between scalar and vectorized versions."""

    miner_vec = miner_hidden[pos]
    validator_vec = validator_hidden[pos]

    # Get top-k indices for miner
    abs_miner = torch.abs(miner_vec)
    topk_result = torch.topk(abs_miner, k=PROOF_TOPK)
    indices = topk_result.indices
    miner_values = miner_vec[indices]

    # Scalar bucketing (original method)
    scalar_buckets = [log_magnitude_bucket(val.item()) for val in miner_values]

    # Vectorized bucketing
    vectorized_buckets = log_magnitude_bucket_vectorized(miner_values).tolist()

    # Compare
    match = scalar_buckets == vectorized_buckets

    print(f"\n=== Bucketing Comparison at position {pos} ===")
    print(f"Top-k indices: {indices[:5].tolist()}...")
    print(f"Values (first 5): {[f'{v:.6f}' for v in miner_values[:5].tolist()]}")
    print(f"Scalar buckets (first 5): {scalar_buckets[:5]}")
    print(f"Vectorized buckets (first 5): {vectorized_buckets[:5]}")
    print(f"Buckets match: {match}")

    if not match:
        for i, (s, v) in enumerate(zip(scalar_buckets, vectorized_buckets)):
            if s != v:
                print(f"  Mismatch at index {i}: scalar={s}, vectorized={v}, value={miner_values[i].item():.10f}")

    return match


def main():
    parser = argparse.ArgumentParser(description="Local proof verification test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Prompt to generate completion from",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens to generate (default: 64)",
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.device)

    # Generate a sample completion
    logger.info(f"Generating completion for prompt: {args.prompt[:50]}...")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    prompt_len = inputs.input_ids.size(1)

    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Get full token sequence
    token_ids = outputs[0].tolist()
    completion_len = len(token_ids) - prompt_len

    logger.info(f"Generated sequence: prompt_len={prompt_len}, completion_len={completion_len}, total={len(token_ids)}")

    # Decode for display
    completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    logger.info(f"Completion: {completion[:100]}...")

    # Extract hidden states using BOTH methods
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING HIDDEN STATES")
    logger.info("=" * 60)

    miner_hidden = extract_hidden_states_miner_style(model, token_ids, args.device)
    validator_hidden = extract_hidden_states_validator_style(model, token_ids, args.device)

    logger.info(f"Miner hidden shape: {miner_hidden.shape}")
    logger.info(f"Validator hidden shape: {validator_hidden.shape}")

    # Compare hidden states
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING HIDDEN STATES")
    logger.info("=" * 60)

    comparison = compare_hidden_states(miner_hidden, validator_hidden)

    print(f"\nSeq len: {comparison['seq_len']}")
    print(f"Hidden dim: {comparison['hidden_dim']}")
    print(f"Identical: {comparison['identical']}")
    print(f"Max diff: {comparison['max_diff']:.2e}")

    print("\nPosition-by-position results:")
    for pos_result in comparison["position_results"]:
        status = "✓" if pos_result["exact_match"] else "✗"
        print(
            f"  {status} pos={pos_result['position']:4d} | "
            f"exact={pos_result['exact_match']} | "
            f"max_diff={pos_result['max_diff']:.2e} | "
            f"miner_norm={pos_result['miner_norm']:.4f} | "
            f"validator_norm={pos_result['validator_norm']:.4f}"
        )

    # Compare sketches
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING SKETCH VERIFICATION")
    logger.info("=" * 60)

    # Initialize verifier
    hidden_dim = model.config.hidden_size
    verifier = GRAILVerifier(hidden_dim=hidden_dim)

    # Generate a fake randomness (in production this comes from beacon)
    import hashlib
    fake_randomness = hashlib.sha256(b"test_randomness").hexdigest()
    r_vec = verifier.generate_r_vec(fake_randomness)

    # Check positions in completion region
    positions_to_check = list(range(prompt_len, min(prompt_len + 16, len(token_ids))))

    sketch_results = compare_sketches(
        miner_hidden, validator_hidden, r_vec, positions_to_check, verifier
    )

    print(f"\nSketch verification: {'ALL PASS ✓' if sketch_results['all_pass'] else 'FAILURES ✗'}")
    print("\nPosition-by-position sketch results:")
    for pos_result in sketch_results["positions"]:
        status = "✓" if pos_result["is_valid"] else "✗"
        indices_status = "same" if pos_result["indices_match"] else "DIFFERENT"
        print(
            f"  {status} pos={pos_result['position']:4d} | "
            f"miner_sketch={pos_result['miner_sketch']:8d} | "
            f"validator_sketch={pos_result['validator_sketch']:8d} | "
            f"diff={pos_result['sketch_diff']:4d} | "
            f"tol={pos_result['tolerance']:4d} | "
            f"indices={indices_status}"
        )
        if not pos_result["indices_match"]:
            print(f"      miner_indices:     {pos_result['miner_indices']}")
            print(f"      validator_indices: {pos_result['validator_indices']}")

    # Compare bucketing methods
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING BUCKETING METHODS")
    logger.info("=" * 60)

    for pos in [prompt_len, prompt_len + 5, prompt_len + 10]:
        if pos < len(token_ids):
            compare_bucketing(miner_hidden, validator_hidden, pos)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if comparison["identical"] and sketch_results["all_pass"]:
        print("\n✓ ALL TESTS PASSED")
        print("  Hidden states are identical between miner and validator methods")
        print("  Sketch verification passes for all checked positions")
    else:
        print("\n✗ TESTS FAILED")
        if not comparison["identical"]:
            print(f"  Hidden states differ: max_diff={comparison['max_diff']:.2e}")
        if not sketch_results["all_pass"]:
            failed = [p for p in sketch_results["positions"] if not p["is_valid"]]
            print(f"  Sketch verification failed at {len(failed)} positions")
            for f in failed[:3]:
                print(f"    - pos={f['position']}: diff={f['sketch_diff']} > tol={f['tolerance']}")


if __name__ == "__main__":
    main()
