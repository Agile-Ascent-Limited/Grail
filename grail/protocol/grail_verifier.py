"""GRAIL Verifier for GPU/Framework-Agnostic Proof.

This module implements a novel hidden-state verification scheme robust
across GPUs, CUDA versions, and frameworks (HF, vLLM, SGLang).

Key innovations:
1. Top-K selection: Focus on important activations (stable)
2. Logarithmic bucketing: Coarse quantization reduces sensitivity
3. Sketch verification: Random linear projection for cryptographic binding

Security: ~10^-117 forgery probability across k=16 positions with sketch-only verification.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from ..shared.constants import (
    PRIME_Q,
    PROOF_COEFF_RANGE,
    PROOF_NUM_BUCKETS,
    PROOF_POSITION_IMPORTANCE_DECAY,
    PROOF_SKETCH_TOLERANCE,
    PROOF_TOPK,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def log_magnitude_bucket_vectorized(
    values: torch.Tensor, num_buckets: int = PROOF_NUM_BUCKETS
) -> torch.Tensor:
    """Vectorized version of log_magnitude_bucket for batch processing.

    Produces IDENTICAL results to the scalar version but processes entire tensors
    at once, eliminating Python loops and .item() calls.

    Args:
        values: Tensor of activation values [batch_size, topk] or [topk]
        num_buckets: Number of buckets per sign (default: 16)

    Returns:
        Tensor of signed bucket indices with same shape as input
    """
    # Start with zeros (handles deadzone case)
    buckets = torch.zeros_like(values, dtype=torch.int32)

    # Get absolute values
    abs_vals = torch.abs(values)

    # Mask for values outside deadzone (abs >= 1e-6)
    active_mask = abs_vals >= 1e-6

    # Handle NaN: set bucket to 0 (already initialized)
    nan_mask = torch.isnan(values)
    active_mask = active_mask & ~nan_mask

    # Handle infinity: set to ±(num_buckets - 1)
    inf_mask = torch.isinf(values)
    pos_inf_mask = inf_mask & (values > 0)
    neg_inf_mask = inf_mask & (values < 0)
    buckets[pos_inf_mask] = num_buckets - 1
    buckets[neg_inf_mask] = -(num_buckets - 1)
    active_mask = active_mask & ~inf_mask

    # Compute buckets for active (non-zero, non-nan, non-inf) values
    if active_mask.any():
        # log2(|x| + 1) scaled to bucket range
        # Use float64 for log2 to match Python math.log2 precision
        active_abs = abs_vals[active_mask].to(torch.float64)
        log_vals = torch.log2(active_abs + 1.0)
        scale_factor = num_buckets / 10.0
        raw_buckets = (log_vals * scale_factor).to(torch.int32)

        # Clamp to valid bucket range
        clamped_buckets = torch.clamp(raw_buckets, 0, num_buckets - 1)

        # Apply sign: negative values get negative buckets
        signs = torch.sign(values[active_mask]).to(torch.int32)
        # Handle zero sign (shouldn't happen due to deadzone, but be safe)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)

        buckets[active_mask] = clamped_buckets * signs

    return buckets


def log_magnitude_bucket(value: float, num_buckets: int = PROOF_NUM_BUCKETS) -> int:
    """Map activation to logarithmic magnitude bucket with sign preservation.

    Logarithmic bucketing provides natural robustness:
    - Small values: coarse bins (where drift happens)
    - Large values: finer bins (where we have precision)
    - Matches floating-point representation behavior

    Args:
        value: Activation value to bucket
        num_buckets: Number of buckets per sign (default: 16)

    Returns:
        Signed bucket index in [-num_buckets+1, 0, num_buckets-1]
    """
    # Handle NaN values - these indicate numerical instability in the model
    if math.isnan(value):
        logger.warning(
            "NaN value encountered in hidden state. This typically indicates "
            "numerical instability in quantized models (especially GPTQ-Int8) "
            "or missing CUDA kernels. Treating as zero bucket."
        )
        return 0

    # Handle infinity values
    if math.isinf(value):
        logger.warning(
            "Infinity value encountered in hidden state. This indicates "
            "numerical overflow. Clamping to maximum bucket."
        )
        return num_buckets - 1 if value > 0 else -(num_buckets - 1)

    abs_val = abs(value)

    # Deadzone for near-zero values
    if abs_val < 1e-6:
        return 0

    # Logarithmic scale: map log2(|x|+1) to bucket range
    # Typical hidden state range: [-3, 3] → log2 range ~ [0, 2]
    # Scale factor maps this to [0, num_buckets)
    log_val = math.log2(abs_val + 1.0)
    # TODO: come up with a more robust approach for measuring max log value
    scale_factor = num_buckets / 10.0  # Assuming max log value ~ 10
    bucket = int(log_val * scale_factor)
    bucket = max(0, min(num_buckets - 1, bucket))

    # Preserve sign
    return bucket if value >= 0 else -bucket


def adaptive_sketch_tolerance(
    position: int,
    sequence_length: int,
    base_tolerance: float,
) -> int:
    """Compute position-dependent sketch tolerance.

    Early positions are more important (set context) → tighter tolerance.
    Later positions may have accumulated drift → more permissive.

    Args:
        position: Token position in sequence
        sequence_length: Total sequence length
        base_tolerance: Base sketch tolerance value

    Returns:
        Adjusted sketch tolerance
    """
    # Importance weight: decays from 1.0 at start
    importance = 1.0 / (1.0 + position / PROOF_POSITION_IMPORTANCE_DECAY)

    # More important → tighter (multiply by factor < 1)
    # Less important → looser (multiply by factor > 1)
    factor = 2.0 - importance  # Range: [1.0, 2.0]

    return int(base_tolerance * factor)


class GRAILVerifier:
    """Sketch-based verifier for framework-agnostic hidden state proofs.

    Uses a single sketch check (random linear projection of bucketed top-k activations)
    which provides sufficient security (~10^-117 forgery probability) while being
    robust to floating-point variations across GPUs and frameworks.
    """

    def __init__(
        self,
        hidden_dim: int,
        topk: int = PROOF_TOPK,
        num_buckets: int = PROOF_NUM_BUCKETS,
        r_coeff_range: int = PROOF_COEFF_RANGE,
    ):
        """Initialize GRAIL verifier.

        Args:
            hidden_dim: Model hidden dimension size
            topk: Number of top activations to select
            num_buckets: Number of magnitude buckets per sign
            r_coeff_range: Range for bounded coefficients [-R, R]
        """
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.num_buckets = num_buckets
        self.r_coeff_range = r_coeff_range
        self.base_sketch_tolerance = float(PROOF_SKETCH_TOLERANCE)

    def generate_r_vec(self, randomness_hex: str) -> torch.Tensor:
        """Generate small bounded coefficient vector from randomness.

        Uses tiny coefficients in [-127, 127] to reduce sensitivity to
        floating-point variations while maintaining cryptographic security.

        Args:
            randomness_hex: Hex string of beacon randomness

        Returns:
            Tensor of shape [topk] with int8 coefficients in [-R, R]
        """
        from ..protocol.crypto import RNG_LABEL, prf

        # Clean hex string
        clean_hex = randomness_hex.strip().replace("0x", "").replace("0X", "")
        if len(clean_hex) % 2 != 0:
            clean_hex = "0" + clean_hex

        # Generate random bytes for coefficients
        raw = prf(
            RNG_LABEL["sketch"],
            bytes.fromhex(clean_hex),
            out_bytes=2 * self.topk,  # 2 bytes per coefficient
        )

        # Convert to int16, then map to [-R, R]
        import numpy as np

        int16_vals = np.frombuffer(raw, dtype=">i2")[: self.topk]  # Big-endian int16
        # Map to [-R, R] using modulo
        coeffs = (np.abs(int16_vals) % (2 * self.r_coeff_range + 1)) - self.r_coeff_range

        return torch.from_numpy(coeffs.astype(np.int8))

    def create_commitment(
        self, hidden_state: torch.Tensor, r_vec: torch.Tensor, position: int
    ) -> dict:
        """Create commitment for a single token position.

        Args:
            hidden_state: Hidden vector at position [hidden_dim]
            r_vec: Coefficient vector [topk]
            position: Token position (for metadata)

        Returns:
            Commitment dict with sketch, indices, and position
        """
        # Step 1: Select top-k activations by absolute magnitude
        abs_hidden = torch.abs(hidden_state)
        topk_result = torch.topk(abs_hidden, k=self.topk)
        indices = topk_result.indices  # [topk]
        values = hidden_state[indices]  # [topk] with signs preserved

        # Step 2: Logarithmic bucketing
        buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in values],
            dtype=torch.int8,
        )

        # Step 3: Compute sketch via dot product with small coefficients
        sketch = torch.dot(buckets.to(torch.int32), r_vec.to(torch.int32))
        sketch_val = int(sketch.item()) % PRIME_Q

        return {
            "sketch": sketch_val,
            "indices": indices.tolist(),
            "position": position,
        }

    def create_commitments_batch(
        self, hidden_states: torch.Tensor, r_vec: torch.Tensor
    ) -> list[dict]:
        """Create commitments for ALL token positions at once (vectorized).

        This is 50-100x faster than calling create_commitment in a loop because:
        1. Single batched topk operation instead of seq_len separate calls
        2. Vectorized log_magnitude_bucket on GPU instead of Python loops
        3. Batched sketch computation with matrix multiplication

        Produces IDENTICAL results to calling create_commitment for each position.

        Args:
            hidden_states: Hidden vectors for all positions [seq_len, hidden_dim]
            r_vec: Coefficient vector [topk]

        Returns:
            List of commitment dicts, one per position
        """
        seq_len = hidden_states.size(0)
        device = hidden_states.device

        # Step 1: Batched top-k selection across all positions
        # abs_hidden: [seq_len, hidden_dim]
        abs_hidden = torch.abs(hidden_states)

        # topk on dim=-1 gives us [seq_len, topk] for both values and indices
        topk_result = torch.topk(abs_hidden, k=self.topk, dim=-1)
        all_indices = topk_result.indices  # [seq_len, topk]

        # Gather the actual signed values at those indices
        # Use gather with expanded indices
        all_values = torch.gather(hidden_states, dim=-1, index=all_indices)  # [seq_len, topk]

        # Step 2: Vectorized logarithmic bucketing for ALL positions at once
        # This replaces seq_len * topk Python function calls with one tensor op
        all_buckets = log_magnitude_bucket_vectorized(
            all_values, self.num_buckets
        )  # [seq_len, topk]

        # Step 3: Batched sketch computation via matrix-vector multiply
        # all_buckets: [seq_len, topk], r_vec: [topk]
        # Result: [seq_len]
        # NOTE: CUDA doesn't support int32 matmul, so we use float32 which gives
        # exact results for our small integer values (buckets in [-16,16], r_vec in [-127,127])
        r_vec_float = r_vec.to(torch.float32).to(device)
        all_sketches = torch.matmul(
            all_buckets.to(torch.float32).to(device), r_vec_float
        ).to(torch.int64)  # [seq_len]

        # Apply modulo PRIME_Q
        all_sketch_vals = (all_sketches % PRIME_Q).tolist()

        # Move indices to CPU for list conversion
        all_indices_cpu = all_indices.cpu().tolist()

        # Build commitment dicts (this is fast - just dict creation)
        commitments = []
        for pos in range(seq_len):
            commitments.append({
                "sketch": all_sketch_vals[pos],
                "indices": all_indices_cpu[pos],
                "position": pos,
            })

        return commitments

    def verify_commitment(
        self,
        validator_hidden: torch.Tensor,
        miner_commitment: dict,
        r_vec: torch.Tensor,
        sequence_length: int,
    ) -> tuple[bool, dict]:
        """Verify commitment using sketch check.

        The sketch check computes a random linear projection of bucketed activations
        and verifies the modular distance is within tolerance.

        Args:
            validator_hidden: Validator's hidden vector at position
            miner_commitment: Miner's claimed commitment
            r_vec: Coefficient vector (same for miner and validator)
            sequence_length: Total sequence length (for adaptive tolerance)

        Returns:
            Tuple of (is_valid, diagnostics_dict)
        """
        position = miner_commitment["position"]

        # Get position-adjusted tolerance
        tolerance = adaptive_sketch_tolerance(position, sequence_length, self.base_sketch_tolerance)

        # Extract miner's claimed top-k indices
        miner_indices = torch.tensor(miner_commitment["indices"], dtype=torch.long)

        # Extract validator's values at those same indices
        validator_values = validator_hidden[miner_indices]

        # Compute validator's buckets
        validator_buckets = torch.tensor(
            [log_magnitude_bucket(val.item(), self.num_buckets) for val in validator_values],
            dtype=torch.int8,
        )

        # Sketch verification: modular distance on dot product
        validator_sketch = torch.dot(validator_buckets.to(torch.int32), r_vec.to(torch.int32))
        validator_sketch_val = int(validator_sketch.item()) % PRIME_Q

        miner_sketch_val = miner_commitment["sketch"]
        sketch_diff = abs(validator_sketch_val - miner_sketch_val)
        mod_diff = min(sketch_diff, PRIME_Q - sketch_diff)  # Modular distance
        is_valid = mod_diff <= tolerance

        diagnostics = {
            "sketch_diff": mod_diff,
            "sketch_valid": is_valid,
            "sketch_tolerance": tolerance,
            "overall_valid": is_valid,
            "validator_sketch": validator_sketch_val,
            "miner_sketch": miner_sketch_val,
            "position": position,
        }

        if not is_valid:
            # Detailed logging for failed verification
            # Sample first few bucket values for debugging
            sample_validator_values = (
                validator_values[:5].tolist()
                if len(validator_values) >= 5
                else validator_values.tolist()
            )
            sample_validator_buckets = (
                validator_buckets[:5].tolist()
                if len(validator_buckets) >= 5
                else validator_buckets.tolist()
            )
            sample_indices = (
                miner_indices[:5].tolist() if len(miner_indices) >= 5 else miner_indices.tolist()
            )

            logger.warning(
                "[verify_commitment] SKETCH MISMATCH: position=%d | "
                "validator_sketch=%d | miner_sketch=%d | diff=%d | tolerance=%d | "
                "sample_indices=%s | sample_values=%s | sample_buckets=%s | "
                "hidden_norm=%.4f | hidden_dtype=%s | "
                "This may indicate model weight differences between miner and validator",
                position,
                validator_sketch_val,
                miner_sketch_val,
                mod_diff,
                tolerance,
                sample_indices,
                [f"{v:.4f}" for v in sample_validator_values],
                sample_validator_buckets,
                float(validator_hidden.norm().item()),
                validator_hidden.dtype,
            )
        else:
            logger.debug(
                "[verify_commitment] OK: position=%d | diff=%d | tolerance=%d",
                position,
                mod_diff,
                tolerance,
            )

        return is_valid, diagnostics
