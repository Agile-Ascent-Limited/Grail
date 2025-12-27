#!/usr/bin/env python3
"""Debug proof mismatches by analyzing a rollout file.

This script loads an existing rollout (from the staging area or cache) and
re-computes the commitments to see exactly where mismatches occur.

Usage:
    # Debug a specific rollout file
    python scripts/debug_proof_mismatch.py --rollout /path/to/rollout.json --checkpoint /path/to/checkpoint

    # Or debug the most recent rollout
    python scripts/debug_proof_mismatch.py --latest --checkpoint /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json
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
)
from grail.shared.constants import LAYER_INDEX, PROOF_TOPK

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load model from checkpoint."""
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading model from: {checkpoint_path}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Model loaded: {model.config.name_or_path}")
    logger.info(f"  Hidden size: {model.config.hidden_size}")
    logger.info(f"  Num layers: {model.config.num_hidden_layers}")
    logger.info(f"  LAYER_INDEX: {LAYER_INDEX}")

    return model


def find_latest_rollout(cache_dir: str = "/root/.cache/grail") -> Path | None:
    """Find the most recent rollout file in staging or cache."""
    cache_path = Path(cache_dir)

    # Check staging directories
    staging_patterns = ["staging_*", "rollouts_*"]
    all_rollouts = []

    for pattern in staging_patterns:
        for staging_dir in cache_path.glob(pattern):
            for rollout_file in staging_dir.glob("*.json"):
                all_rollouts.append(rollout_file)

    if not all_rollouts:
        logger.warning(f"No rollout files found in {cache_dir}")
        return None

    # Sort by modification time
    latest = max(all_rollouts, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest rollout: {latest}")
    return latest


def analyze_rollout(rollout_path: Path, model, device: str = "cuda:0"):
    """Analyze a rollout file and compare commitments."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {rollout_path}")
    logger.info(f"{'='*60}")

    with open(rollout_path) as f:
        rollout_data = json.load(f)

    # Extract key fields
    tokens = rollout_data.get("tokens", [])
    commitments = rollout_data.get("commitments", [])
    beacon = rollout_data.get("beacon", {})
    rollout_info = rollout_data.get("rollout", {})
    model_info = rollout_data.get("model", {})

    prompt_len = rollout_info.get("prompt_length", 0)
    completion_len = rollout_info.get("completion_length", 0)
    randomness_hex = beacon.get("randomness", "")

    logger.info(f"Tokens: {len(tokens)} (prompt={prompt_len}, completion={completion_len})")
    logger.info(f"Commitments: {len(commitments)}")
    logger.info(f"Model claimed: {model_info.get('name', 'unknown')}")
    logger.info(f"Checkpoint window: {model_info.get('checkpoint_window', 'unknown')}")
    logger.info(f"Beacon randomness: {randomness_hex[:32]}...")

    if len(tokens) != len(commitments):
        logger.error(f"MISMATCH: tokens={len(tokens)} != commitments={len(commitments)}")
        return

    # Initialize verifier
    hidden_dim = model.config.hidden_size
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness_hex)

    logger.info(f"\nR-vector (first 8): {r_vec[:8].tolist()}")

    # Run model forward pass (same as validator)
    logger.info("\nRunning model forward pass...")
    full_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.inference_mode():
        # EXACTLY like validator - no attention_mask, no position_ids
        outs = model(full_ids, output_hidden_states=True)

    h_layer = outs.hidden_states[LAYER_INDEX][0]  # [seq_len, hidden_dim]
    logger.info(f"Hidden states shape: {h_layer.shape}")
    logger.info(f"Hidden states dtype: {h_layer.dtype}")

    # Compare commitments
    logger.info(f"\n{'='*60}")
    logger.info("COMMITMENT COMPARISON")
    logger.info(f"{'='*60}")

    mismatches = []
    sample_positions = [0, prompt_len - 1, prompt_len, prompt_len + 5, len(tokens) - 1]
    sample_positions = [p for p in sample_positions if 0 <= p < len(tokens)]

    for pos in sample_positions:
        miner_commitment = commitments[pos]

        # Recompute commitment locally
        local_commitment = verifier.create_commitment(h_layer[pos], r_vec, pos)

        # Also verify the miner's commitment against our hidden state
        is_valid, diagnostics = verifier.verify_commitment(
            h_layer[pos], miner_commitment, r_vec, len(tokens)
        )

        # Extract details
        miner_sketch = miner_commitment.get("sketch", -1)
        local_sketch = local_commitment["sketch"]
        miner_indices = miner_commitment.get("indices", [])
        local_indices = local_commitment["indices"]

        sketch_diff = abs(miner_sketch - local_sketch)
        indices_match = miner_indices == local_indices

        status = "✓" if is_valid else "✗"

        logger.info(f"\nPosition {pos} ({status}):")
        logger.info(f"  Token ID: {tokens[pos]}")
        logger.info(f"  Hidden norm: {h_layer[pos].norm().item():.6f}")
        logger.info(f"  Miner sketch:    {miner_sketch}")
        logger.info(f"  Local sketch:    {local_sketch}")
        logger.info(f"  Sketch diff:     {sketch_diff}")
        logger.info(f"  Tolerance:       {diagnostics['sketch_tolerance']}")
        logger.info(f"  Indices match:   {indices_match}")
        logger.info(f"  Miner indices (first 5):  {miner_indices[:5]}")
        logger.info(f"  Local indices (first 5):  {local_indices[:5]}")

        if not is_valid:
            mismatches.append(pos)

            # Deep dive into the mismatch
            logger.info(f"\n  DETAILED ANALYSIS at pos {pos}:")

            # Get values at miner's indices
            miner_idx_tensor = torch.tensor(miner_indices, dtype=torch.long, device=device)
            values_at_miner_indices = h_layer[pos][miner_idx_tensor]

            # Compute buckets for miner's indices
            miner_buckets = [
                log_magnitude_bucket(val.item())
                for val in values_at_miner_indices
            ]

            # Also get local top-k values
            abs_hidden = torch.abs(h_layer[pos])
            topk_result = torch.topk(abs_hidden, k=PROOF_TOPK)
            local_values = h_layer[pos][topk_result.indices]
            local_buckets = [log_magnitude_bucket(val.item()) for val in local_values]

            logger.info(f"  Miner indices values (first 5): {[f'{v:.6f}' for v in values_at_miner_indices[:5].tolist()]}")
            logger.info(f"  Miner buckets (first 5):        {miner_buckets[:5]}")
            logger.info(f"  Local values (first 5):         {[f'{v:.6f}' for v in local_values[:5].tolist()]}")
            logger.info(f"  Local buckets (first 5):        {local_buckets[:5]}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    if not mismatches:
        logger.info("✓ All sample positions match!")
        logger.info("  If the validator is still failing, check:")
        logger.info("    1. Checkpoint window matches")
        logger.info("    2. Beacon randomness is current")
        logger.info("    3. Challenge indices (k=16) might hit other positions")
    else:
        logger.error(f"✗ {len(mismatches)} mismatches found at positions: {mismatches}")
        logger.info("  Possible causes:")
        logger.info("    1. Different model checkpoint (most likely)")
        logger.info("    2. Tokenization differences")
        logger.info("    3. Data type differences (bfloat16 vs float16)")


def main():
    parser = argparse.ArgumentParser(description="Debug proof mismatches")
    parser.add_argument(
        "--rollout",
        type=str,
        help="Path to rollout JSON file",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent rollout file",
    )
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
        "--cache-dir",
        type=str,
        default="/root/.cache/grail",
        help="Cache directory to search for rollouts",
    )
    args = parser.parse_args()

    # Find rollout file
    if args.rollout:
        rollout_path = Path(args.rollout)
    elif args.latest:
        rollout_path = find_latest_rollout(args.cache_dir)
    else:
        parser.error("Either --rollout or --latest must be specified")

    if rollout_path is None or not rollout_path.exists():
        logger.error(f"Rollout file not found: {rollout_path}")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Analyze
    analyze_rollout(rollout_path, model, args.device)


if __name__ == "__main__":
    main()
