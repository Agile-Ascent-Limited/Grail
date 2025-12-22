#!/usr/bin/env python3
"""
Self-validation script for GRAIL parquet files.

Reverse-engineers the validator's logic to verify rollouts will pass env_prompt_valid
before uploading. Use this to debug multi-GPU aggregation issues.

Features:
- Validates rollout prompts match expected seeds (env_prompt_valid check)
- Detects DUPLICATE NONCES (same rollout_group at non-contiguous file positions)
- Detects gaps in problem indices
- Reports rollout ordering and group statistics

Usage:
    # Validate a local parquet file
    python scripts/validate_parquet.py /path/to/file.parquet

    # Validate with verbose output (shows each rollout)
    python scripts/validate_parquet.py /path/to/file.parquet -v

    # Validate and show first N failures in detail
    python scripts/validate_parquet.py /path/to/file.parquet --show-failures 5

    # Download from R2 and validate
    python scripts/validate_parquet.py --hotkey 5Gxxx... --window 7155540

Duplicate Nonce Detection:
    If you see "Duplicate nonce X; invalidating uid" from the validator but this
    script says "all ok", check the "Checking for duplicate nonces" section.

    Duplicate nonces occur when stale staging files from a previous window get
    mixed with current rollouts. The script detects this by finding rollout_groups
    that appear at non-contiguous file positions.

    Fix: pm2 stop all && rm -rf ~/.cache/grail/.worker-barrier/*.json && pm2 start all
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Add grail to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def derive_env_seed(wallet_addr: str, window_hash: str, problem_index: int) -> int:
    """
    Derive canonical environment seed - EXACT copy of validator logic.

    This is the critical function that determines which problem should be
    at each file position. The validator uses this same formula.

    Args:
        wallet_addr: Miner's SS58 wallet address (hotkey)
        window_hash: Block hash at window start (from blockchain)
        problem_index: File position (0, 1, 2, ...) - NOT rollout_group!

    Returns:
        Integer seed for environment reset
    """
    try:
        idx = int(problem_index)
    except Exception:
        idx = 0

    material = f"{wallet_addr}:{window_hash}:{idx}".encode()
    seed_hex = hashlib.sha256(b"seed|" + material).hexdigest()
    return int(seed_hex[:8], 16)


def load_parquet_file(filepath: str) -> dict:
    """Load and parse a parquet file."""
    import pyarrow.parquet as pq

    from grail.infrastructure.parquet_io import deserialize_parquet_to_window

    with open(filepath, "rb") as f:
        data = f.read()

    return deserialize_parquet_to_window(data)


def setup_tokenizer_with_chat_template(tokenizer):
    """
    Set up the tokenizer with GRAIL's custom chat template.

    CRITICAL: The validator loads the chat template from the checkpoint directory
    (chat_template.jinja). This template includes the SYSTEM_PROMPT which gets
    injected into every prompt. We must replicate this for validation to match.
    """
    from grail.shared.chat_templates import build_qwen_chat_template

    tokenizer.chat_template = build_qwen_chat_template()
    return tokenizer


def get_prompt_for_seed(seed: int, tokenizer) -> list[int]:
    """
    Reconstruct the expected prompt token IDs for a given seed.

    Uses the same environment adapter as the validator.
    """
    from grail.environments.registry import get_adapter
    from grail.shared.constants import CURRENT_ENV_ID

    adapter = get_adapter(CURRENT_ENV_ID)
    return adapter.build_prompt_ids(seed, tokenizer)


def validate_rollout(
    inference: dict,
    file_position: int,
    hotkey: str,
    window_hash: str,
    tokenizer,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Validate a single rollout against the expected prompt.

    Args:
        inference: The inference/rollout dict from the parquet file
        file_position: Position in file (0, 1, 2, ...) - this becomes group_index
        hotkey: Miner's hotkey (SS58 address)
        window_hash: Window's block hash
        tokenizer: The tokenizer to use
        verbose: Print detailed info

    Returns:
        (is_valid, error_message)
    """
    commit = inference.get("commit", {})
    rollout = commit.get("rollout", {})
    tokens = commit.get("tokens", [])
    prompt_length = rollout.get("prompt_length", 0)
    rollout_group = inference.get("rollout_group", 0)

    # Derive the seed the validator will use (based on FILE POSITION, not rollout_group!)
    validator_seed = derive_env_seed(hotkey, window_hash, file_position)

    # Get expected prompt tokens
    expected_prompt = get_prompt_for_seed(validator_seed, tokenizer)

    # Get actual prompt tokens from rollout
    actual_prompt = tokens[:prompt_length]

    # Check 1: Prompt length match
    if len(expected_prompt) != prompt_length:
        # Decode full prompts for debugging
        expected_text = tokenizer.decode(expected_prompt, skip_special_tokens=False)
        actual_text = tokenizer.decode(actual_prompt, skip_special_tokens=False) if actual_prompt else "(empty)"
        return False, (
            f"Prompt length mismatch: expected={len(expected_prompt)}, "
            f"actual={prompt_length} (file_pos={file_position}, rollout_group={rollout_group}, seed={validator_seed})\n"
            f"      Expected:\n{expected_text}\n"
            f"      Actual:\n{actual_text}"
        )

    # Check 2: Prompt tokens match exactly
    if actual_prompt != expected_prompt:
        # Find first mismatch
        mismatch_pos = -1
        for i in range(min(len(actual_prompt), len(expected_prompt))):
            if actual_prompt[i] != expected_prompt[i]:
                mismatch_pos = i
                break

        return False, (
            f"Token mismatch at position {mismatch_pos}: "
            f"expected={expected_prompt[mismatch_pos] if mismatch_pos >= 0 else '?'}, "
            f"actual={actual_prompt[mismatch_pos] if mismatch_pos >= 0 else '?'} "
            f"(file_pos={file_position}, rollout_group={rollout_group}, seed={validator_seed})"
        )

    if verbose:
        print(f"  [OK] file_pos={file_position}, rollout_group={rollout_group}, seed={validator_seed}")

    return True, ""


def validate_parquet_file(
    filepath: str,
    window_hash: str | None = None,
    verbose: bool = False,
    show_failures: int = 0,
    model_name: str | None = None,
) -> tuple[int, int, list[str]]:
    """
    Validate all rollouts in a parquet file.

    Args:
        filepath: Path to the parquet file
        window_hash: Override window hash (otherwise fetched from chain)
        verbose: Print each rollout result
        show_failures: Number of failure details to show
        model_name: Override model name for tokenizer (otherwise extracted from parquet)

    Returns:
        (valid_count, total_count, failure_messages)
    """
    from transformers import AutoTokenizer

    print(f"Loading parquet file: {filepath}")
    window_data = load_parquet_file(filepath)

    hotkey = window_data.get("wallet", "")
    window_start = window_data.get("window_start", 0)
    inferences = window_data.get("inferences", [])

    print(f"  Hotkey: {hotkey}")
    print(f"  Window: {window_start}")
    print(f"  Rollouts: {len(inferences)}")

    # Show model info from first rollout
    parquet_model = ""
    if inferences:
        first_commit = inferences[0].get("commit", {})
        model_info = first_commit.get("model", {})
        parquet_model = model_info.get("name", "")
        print(f"  Model in parquet: {parquet_model or '(not set)'}")

    # Get window hash from chain if not provided
    if not window_hash:
        print("\nFetching window hash from chain...")
        try:
            import bittensor as bt
            subtensor = bt.subtensor(network="finney")
            block_hash = subtensor.get_block_hash(window_start)
            window_hash = block_hash
            print(f"  Window hash: {window_hash[:20]}...")
        except Exception as e:
            print(f"  ERROR: Could not fetch window hash: {e}")
            print("  Please provide --window-hash manually")
            return 0, len(inferences), [f"Missing window hash: {e}"]

    # Determine model name for tokenizer
    if not model_name:
        # Try to use model name from parquet, but only if it looks like a valid HF repo
        # (contains '/' like "Qwen/Qwen3-4B-Instruct-2507")
        if parquet_model and "/" in parquet_model:
            model_name = parquet_model
        else:
            # Fall back to default - parquet has local checkpoint name or is empty
            model_name = "Qwen/Qwen3-4B-Instruct-2507"  # Current GRAIL default
            if parquet_model:
                print(f"  Note: Parquet model '{parquet_model}' is not a HF repo, using default: {model_name}")
            else:
                print(f"  Warning: No model name in parquet, using default: {model_name}")

    # Load tokenizer
    print(f"\nLoading tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Apply GRAIL's custom chat template (with SYSTEM_PROMPT injection)
    # This matches what the validator does when loading from checkpoint
    tokenizer = setup_tokenizer_with_chat_template(tokenizer)
    print("  Applied GRAIL chat template with SYSTEM_PROMPT")

    # Group rollouts by rollout_group to check ordering
    print("\nAnalyzing rollout ordering...")
    groups_in_order = []
    current_group = None
    for i, inf in enumerate(inferences):
        group = inf.get("rollout_group", 0)
        if group != current_group:
            groups_in_order.append((i, group))
            current_group = group

    # Check if sorted by rollout_group
    group_values = [g for _, g in groups_in_order]
    is_sorted = group_values == sorted(group_values)

    if not is_sorted:
        print(f"  WARNING: Rollouts are NOT sorted by rollout_group!")
        print(f"  First 10 groups in file order: {group_values[:10]}")
    else:
        print(f"  Rollouts are sorted by rollout_group")
        print(f"  Group range: {min(group_values)} to {max(group_values)}")

    # Check for gaps
    unique_groups = sorted(set(group_values))
    expected_groups = list(range(unique_groups[-1] + 1)) if unique_groups else []
    missing_groups = set(expected_groups) - set(unique_groups)

    if missing_groups:
        first_gap = min(missing_groups)
        print(f"  GAP DETECTED: First missing group is {first_gap}")
        print(f"  Missing groups: {sorted(missing_groups)[:10]}{'...' if len(missing_groups) > 10 else ''}")
    else:
        print(f"  No gaps: contiguous from {unique_groups[0]} to {unique_groups[-1]}")

    # Check for DUPLICATE NONCES (exact same nonce value appearing twice)
    # This matches the validator logic in miner_validator.py:_check_inference_consistency
    # nonce = rollout_group * MULTIPLIER + rollout_index, so each rollout should have a unique nonce
    # NOTE: Multiplier was 10 in old code (caused collisions with batch_size>10), now 100
    print("\nChecking for duplicate nonces (validator logic)...")

    # First, show a sample of raw data to verify reading correctly
    print("  Raw data sample (first 50 entries):")
    for i, inf in enumerate(inferences[:50]):
        stored_nonce = inf.get("nonce")
        stored_group = inf.get("rollout_group")
        stored_idx = inf.get("rollout_index")
        print(f"    pos={i}: nonce={stored_nonce}, rollout_group={stored_group}, rollout_index={stored_idx}")

    nonces_seen = {}  # nonce -> list of (file_pos, raw_data) tuples
    for file_pos, inf in enumerate(inferences):
        nonce = inf.get("nonce")
        if nonce is None:
            # Try to compute from rollout_group and rollout_index if nonce field is missing
            group = inf.get("rollout_group", 0)
            idx = inf.get("rollout_index", 0)
            nonce = group * 10 + idx
        if nonce not in nonces_seen:
            nonces_seen[nonce] = []
        nonces_seen[nonce].append((file_pos, {
            "stored_nonce": inf.get("nonce"),
            "rollout_group": inf.get("rollout_group"),
            "rollout_index": inf.get("rollout_index"),
            "window_start": inf.get("window_start"),
        }))

    duplicate_nonces = [(nonce, entries) for nonce, entries in nonces_seen.items() if len(entries) > 1]

    if duplicate_nonces:
        print(f"\n  ⚠️  DUPLICATE NONCES DETECTED: {len(duplicate_nonces)} nonce(s) appear multiple times!")
        print(f"  The validator will reject this file with 'Duplicate nonce X; invalidating uid'")
        print()
        for nonce, entries in sorted(duplicate_nonces)[:10]:  # Show first 10
            print(f"    nonce={nonce}: appears at {len(entries)} file positions")
            for pos, data in entries[:3]:  # Show first 3 occurrences
                print(f"      pos={pos}: stored_nonce={data['stored_nonce']}, "
                      f"group={data['rollout_group']}, idx={data['rollout_index']}, "
                      f"window={data['window_start']}")
            if len(entries) > 3:
                print(f"      ... and {len(entries) - 3} more")
        if len(duplicate_nonces) > 10:
            print(f"    ... and {len(duplicate_nonces) - 10} more duplicate nonces")
        print()
        # Diagnose the cause based on patterns
        # Check if duplicates are from formula collision (different group/idx compute to same nonce)
        formula_collision = False
        for nonce, entries in duplicate_nonces[:5]:
            groups = set(e[1]['rollout_group'] for e in entries)
            if len(groups) > 1:
                formula_collision = True
                break

        if formula_collision:
            print("  CAUSE: Nonce formula collision! Different (group, idx) pairs compute to same nonce.")
            print("         This happens when batch_size > 10 with old nonce formula (group*10 + idx).")
            print("         Example: group=0,idx=10 and group=1,idx=0 both give nonce=10")
            print()
            print("  FIX: Update to latest code where multiplier is 100 instead of 10:")
            print("       rollout_nonce = base_nonce * 100 + rollout_idx")
        else:
            print("  CAUSE: Stale staging files from a previous window got mixed in.")
            print("  FIX: Clear staging directory and restart miners:")
            print("    pm2 stop all")
            print("    rm -rf ~/.cache/grail/.worker-barrier/*.json")
            print("    pm2 start all")
    else:
        print(f"  ✅ No duplicate nonces detected ({len(nonces_seen)} unique nonces)")

    # Validate each rollout
    print("\nValidating rollouts...")
    valid_count = 0
    failures = []

    # Track group_index (file position for each unique group)
    group_index_map = {}  # rollout_group -> group_index (file order)

    for file_pos, inf in enumerate(inferences):
        rollout_group = inf.get("rollout_group", 0)

        # Assign group_index based on first encounter (same as validator)
        if rollout_group not in group_index_map:
            group_index_map[rollout_group] = len(group_index_map)

        group_index = group_index_map[rollout_group]

        is_valid, error = validate_rollout(
            inf,
            group_index,  # Validator uses group_index, not file_pos
            hotkey,
            window_hash,
            tokenizer,
            verbose=verbose,
        )

        if is_valid:
            valid_count += 1
        else:
            failures.append(f"Rollout {file_pos}: {error}")
            if verbose or len(failures) <= show_failures:
                print(f"  [FAIL] {error}")

    return valid_count, len(inferences), failures


def main():
    parser = argparse.ArgumentParser(
        description="Validate GRAIL parquet files against expected prompts"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Path to parquet file to validate",
    )
    parser.add_argument(
        "--hotkey",
        help="Miner hotkey (for R2 download)",
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Window number (for R2 download)",
    )
    parser.add_argument(
        "--window-hash",
        help="Override window block hash (otherwise fetched from chain)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each rollout result",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=5,
        help="Number of failure details to show (default: 5)",
    )
    parser.add_argument(
        "--model-name",
        help="Model name for tokenizer (otherwise extracted from parquet or defaults to Qwen3-4B-Instruct-2507)",
    )

    args = parser.parse_args()

    # Determine filepath
    if args.filepath:
        filepath = args.filepath
    elif args.hotkey and args.window:
        # Download from R2
        print(f"Downloading from R2: {args.hotkey[:12]}...-window-{args.window}.parquet")
        # TODO: Implement R2 download
        print("ERROR: R2 download not yet implemented. Please provide a local file path.")
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Validate
    valid, total, failures = validate_parquet_file(
        filepath,
        window_hash=args.window_hash,
        verbose=args.verbose,
        show_failures=args.show_failures,
        model_name=args.model_name,
    )

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Valid rollouts:   {valid}/{total} ({100*valid/total:.1f}%)" if total else "No rollouts")
    print(f"Failed rollouts:  {total - valid}")

    if failures and not args.verbose:
        print(f"\nFirst {min(len(failures), args.show_failures)} failures:")
        for f in failures[:args.show_failures]:
            print(f"  - {f}")
        if len(failures) > args.show_failures:
            print(f"  ... and {len(failures) - args.show_failures} more")

    if valid == total:
        print("\n✅ All rollouts should pass env_prompt_valid!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - valid} rollouts will FAIL env_prompt_valid")
        sys.exit(1)


if __name__ == "__main__":
    main()
