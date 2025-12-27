#!/usr/bin/env python3
"""Check checkpoint integrity and compute weights hash.

This script verifies that a checkpoint is correct by computing its weights hash
and comparing it to what the trainer published.

Usage:
    python scripts/check_checkpoint.py --checkpoint /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_weights_hash_from_files(checkpoint_path: Path) -> str:
    """Compute weights hash from safetensor files.

    This matches the hash computation in delta_checkpoint.py.
    """
    # Find all safetensor files
    safetensor_files = sorted(checkpoint_path.glob("*.safetensors"))

    if not safetensor_files:
        raise ValueError(f"No safetensor files found in {checkpoint_path}")

    logger.info(f"Found {len(safetensor_files)} safetensor files")

    # Load and hash all tensors
    hasher = hashlib.sha256()

    for sf_path in safetensor_files:
        logger.info(f"  Loading: {sf_path.name}")
        tensors = load_file(str(sf_path))

        # Sort keys for deterministic hashing
        for key in sorted(tensors.keys()):
            tensor = tensors[key]
            # Convert to contiguous bytes
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor_bytes = tensor.contiguous().numpy().tobytes()
            hasher.update(key.encode("utf-8"))
            hasher.update(tensor_bytes)

    return hasher.hexdigest()


def compute_weights_hash_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute weights hash from a loaded state dict.

    Same as delta_checkpoint.py:compute_weights_hash()
    """
    hasher = hashlib.sha256()

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor_bytes = tensor.contiguous().numpy().tobytes()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor_bytes)

    return hasher.hexdigest()


def check_metadata(checkpoint_path: Path) -> dict | None:
    """Load and display checkpoint metadata."""
    meta_path = checkpoint_path / "metadata.json"

    if not meta_path.exists():
        logger.warning(f"No metadata.json found at {meta_path}")
        return None

    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info("\nCheckpoint Metadata:")
    logger.info(f"  Window: {metadata.get('window', 'unknown')}")
    logger.info(f"  Type: {metadata.get('checkpoint_type', 'unknown')}")
    logger.info(f"  Weights hash: {metadata.get('weights_hash', 'none')}")
    logger.info(f"  Model name: {metadata.get('model_name', 'unknown')}")
    logger.info(f"  Created at: {metadata.get('created_at', 'unknown')}")

    if metadata.get("checkpoint_type") == "DELTA":
        logger.info(f"  Prev window: {metadata.get('prev_window', 'unknown')}")
        logger.info(f"  Anchor window: {metadata.get('anchor_window', 'unknown')}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Check checkpoint integrity")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip computing the full weights hash (faster)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Checking checkpoint: {checkpoint_path}")

    # Check metadata
    metadata = check_metadata(checkpoint_path)

    # List files
    logger.info("\nCheckpoint Files:")
    total_size = 0
    for f in sorted(checkpoint_path.iterdir()):
        size = f.stat().st_size
        total_size += size
        logger.info(f"  {f.name}: {size / 1e6:.1f} MB")
    logger.info(f"  TOTAL: {total_size / 1e9:.2f} GB")

    # Compute hash if requested
    if not args.skip_hash:
        logger.info("\nComputing weights hash (this may take a minute)...")
        try:
            computed_hash = compute_weights_hash_from_files(checkpoint_path)
            logger.info(f"\nComputed weights hash: {computed_hash}")

            if metadata and metadata.get("weights_hash"):
                expected_hash = metadata["weights_hash"]
                if computed_hash == expected_hash:
                    logger.info("✓ Hash matches metadata!")
                else:
                    logger.error("✗ Hash MISMATCH!")
                    logger.error(f"  Expected: {expected_hash}")
                    logger.error(f"  Computed: {computed_hash}")
                    sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to compute hash: {e}")
            sys.exit(1)

    # Load model and check hidden dimension
    logger.info("\nLoading model config...")
    try:
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"  Model type: {config.get('model_type', 'unknown')}")
            logger.info(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
            logger.info(f"  Num layers: {config.get('num_hidden_layers', 'unknown')}")
            logger.info(f"  Vocab size: {config.get('vocab_size', 'unknown')}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("CHECKPOINT CHECK COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
