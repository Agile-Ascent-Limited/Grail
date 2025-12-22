"""Environment loop for GRPO rollout generation with GRAIL proofs.

Provides AgentEnvLoop class that:
  - Wraps model/tokenizer with sampling config
  - Runs single-turn episodes with logprob tracking
  - Generates GRAIL proof commitments
  - Supports both sequential and vectorized GRPO group generation
  - Returns GRPORollout dataclass compatible with mining packaging
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

try:
    import bittensor as bt
except Exception:  # pragma: no cover - optional in offline mode
    bt = None  # type: ignore[assignment]
import numpy as np
import torch

from ..shared.constants import GRAIL_PROOF_VERSION, LAYER_INDEX, MAX_NEW_TOKENS
from ..shared.hf_compat import resolve_hidden_size
from .core import ChatMessage, MultiTurnEnv

logger = logging.getLogger(__name__)


def _shutdown_engine_and_free_gpu(engine_ref: Any | None, engine_name: str = "engine") -> None:
    """Shared GPU cleanup logic for inference engines (vLLM, SGLang, etc).

    Args:
        engine_ref: Engine instance to shutdown (will be set to None by caller)
        engine_name: Name for logging (e.g., "vLLM", "SGLang async")
    """
    import gc

    if engine_ref is None:
        return

    try:
        logger.info("Shutting down %s engine...", engine_name)

        # Try graceful shutdown if available
        if hasattr(engine_ref, "shutdown"):
            engine_ref.shutdown()

        # Force garbage collection to release resources
        gc.collect()

        # Synchronize GPU and clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("%s engine shutdown complete", engine_name)
    except Exception as e:
        logger.warning("Error during %s shutdown: %s", engine_name, e)


def _decode_prompts(tokenizer: Any, prompt_ids_batch: list[list[int]]) -> list[str]:
    """Decode batch of token IDs to text prompts.

    Args:
        tokenizer: HuggingFace tokenizer instance
        prompt_ids_batch: List of tokenized prompts

    Returns:
        List of decoded text prompts
    """
    return [tokenizer.decode(p, skip_special_tokens=False) for p in prompt_ids_batch]


def _tokenize_completion(tokenizer: Any, completion_text: str, fallback: list[int]) -> list[int]:
    """Tokenize completion text with error handling.

    Args:
        tokenizer: HuggingFace tokenizer instance
        completion_text: Generated completion text
        fallback: Token IDs to return on error (typically prompt IDs)

    Returns:
        List of completion token IDs
    """
    try:
        comp_ids = tokenizer(completion_text, return_tensors=None, return_attention_mask=False)[
            "input_ids"
        ]

        # Handle nested list structure
        if isinstance(comp_ids, list) and len(comp_ids) > 0 and isinstance(comp_ids[0], list):
            comp_ids = comp_ids[0]

        return cast(list[int], comp_ids if isinstance(comp_ids, list) else [])
    except (AttributeError, IndexError, TypeError, KeyError) as e:
        logger.debug("Failed to tokenize completion: %s", e)
        return fallback


def _set_global_seed(seed: int) -> None:
    """Seed all relevant RNGs for deterministic generation."""
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class GenerationParams:
    """Text generation parameters passed to backends.

    Supports per-sample deterministic generation via seeds when the backend
    implementation allows it (HF via torch.Generator list; vLLM via seed field).
    """

    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int | None = 50
    repetition_penalty: float | None = 1.1
    trim_right_padding: bool = False


class TextGenBackend(Protocol):
    """Abstract interface for batched text generation backends.

    All backends must return tuples: (tokens, chosen_logprobs_or_none).
    The second element may be None when logprobs are not requested or unsupported.
    """

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Generate completions for a batch of prompts.

        Args:
            prompt_ids_batch: Batch of tokenized prompts
            params: Generation parameters (temperature, top_p, etc)
            seeds: Optional per-sample seeds for deterministic sampling

        Returns:
            List of tuples per prompt: (full_sequence, chosen_logprobs_or_none).
            - full_sequence: Token IDs for prompt + completion
            - chosen_logprobs_or_none: List of logprobs for chosen completion tokens,
              or None if not requested/supported

        Backends must left-pad internally and remove left padding before
        returning, so returned sequences are aligned as [prompt + completion].
        If params.trim_right_padding is True, they should trim any trailing pad
        tokens in the completion region as appropriate for the tokenizer.
        """
        ...


class HFBackend:
    """HuggingFace generation backend using a provided model/tokenizer instance.

    This backend does not own the model; it reuses the instance passed in to
    maintain a single copy in memory when used for both generation and proofs.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        *,
        return_chosen_logprobs: bool = False,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._return_chosen_logprobs = bool(return_chosen_logprobs)

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        batch_size = len(prompt_ids_batch)
        if batch_size == 0:
            return []

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        eos_id = self._tokenizer.eos_token_id

        # Left-pad inputs to max length
        max_len = max(len(p) for p in prompt_ids_batch)
        padded_inputs: list[list[int]] = []
        attention_masks: list[list[int]] = []
        left_pads: list[int] = []
        for p in prompt_ids_batch:
            pad_len = max_len - len(p)
            padded_inputs.append([pad_id] * pad_len + p)
            attention_masks.append([0] * pad_len + [1] * len(p))
            left_pads.append(pad_len)

        input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self._device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.max_new_tokens),
            "temperature": float(params.temperature),
            "do_sample": bool(params.do_sample),
            "top_p": float(params.top_p),
            "return_dict_in_generate": True,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        if params.top_k is not None:
            gen_kwargs["top_k"] = int(params.top_k)
        if params.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(params.repetition_penalty)

        # Seeds are currently ignored to favor batched generation efficiency
        if seeds is not None and seeds:
            logger.debug("HFBackend: ignoring seeds for batched generation")

        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        results: list[tuple[list[int], list[float] | None]] = []
        for b in range(batch_size):
            seq = outputs.sequences[b]
            # Remove left padding
            seq_wo_left = seq[left_pads[b] :]
            all_ids = seq_wo_left.tolist()

            if params.trim_right_padding:
                # Trim trailing padding conservatively
                if pad_id is not None and pad_id != eos_id:
                    # Find last non-pad index
                    last_non_pad = len(all_ids) - 1
                    while last_non_pad >= 0 and all_ids[last_non_pad] == pad_id:
                        last_non_pad -= 1
                    all_ids = all_ids[: max(0, last_non_pad + 1)]
            # HF backend doesn't currently support logprobs extraction
            results.append((all_ids, None))

        return results


class VLLMServerBackend:
    """vLLM server (OpenAI-compatible) backend over HTTP with async API.

    Uses AsyncOpenAI client to interact with a running vLLM server. Deterministic
    generation is achieved by passing a per-request seed when provided.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        tokenizer: Any,
        timeout: float = 300.0,
        max_concurrent_requests: int = 32,
        return_chosen_logprobs: bool = False,
        warn_on_missing_token_ids: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._timeout = float(timeout)
        self._max_concurrent_requests = max_concurrent_requests
        self._return_chosen_logprobs = bool(return_chosen_logprobs)
        self._warn_on_missing_token_ids = bool(warn_on_missing_token_ids)

        # Client is created lazily to handle event loop changes
        self._client = None
        self._client_lock = None  # Will be created per-event-loop

    def _get_client(self):
        """Get or create AsyncOpenAI client for current event loop.

        The client is created lazily and recreated if the event loop has changed.
        This handles window transitions where the vLLM server reloads and the
        event loop may be closed/recreated.
        """
        import asyncio

        from openai import AsyncOpenAI

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - will be created when needed
            loop = None

        # Check if we need to create/recreate the client
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="EMPTY",
                timeout=self._timeout,
            )
            logger.debug("Created new AsyncOpenAI client for vLLM backend")

        return self._client

    def _recreate_client(self):
        """Force recreation of the AsyncOpenAI client.

        Called when we detect the event loop has been closed/changed.
        """
        from openai import AsyncOpenAI

        logger.warning("Recreating AsyncOpenAI client due to event loop change")
        self._client = AsyncOpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="EMPTY",
            timeout=self._timeout,
        )

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Generate completions using vLLM server async API.

        Now properly async - called directly from async context without nested event loops.
        """
        return await self._async_generate_batch(prompt_ids_batch, params, seeds)

    async def _async_generate_batch(
        self,
        prompt_ids_batch: list[list[int]],
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        import asyncio
        import time

        batch_start = time.time()
        batch_size = len(prompt_ids_batch)
        logger.info(
            "vLLMServer: Starting ASYNC batch of %d prompts (using prompt_token_ids directly)",
            batch_size,
        )

        # Use configurable semaphore to control client-side concurrency
        sem = asyncio.Semaphore(self._max_concurrent_requests)

        async def _call_one_async(
            idx: int, prompt_ids: list[int], rnd_seed: int | None
        ) -> tuple[int, str, list[float] | None, list[int] | None]:
            """Send prompt token IDs directly to vLLM to avoid decode/re-encode round-trip."""
            max_retries = 3
            base_backoff = 1.0
            async with sem:
                for attempt in range(max_retries):
                    req_start = time.time()
                    try:
                        # Build completion kwargs
                        # Note: prompt is required by OpenAI client but vLLM ignores it
                        # when prompt_token_ids is provided in extra_body
                        completion_kwargs: dict[str, Any] = {
                            "model": self._model_name,
                            "prompt": "",  # Placeholder - vLLM uses prompt_token_ids instead
                            "max_tokens": int(params.max_new_tokens),
                            "temperature": float(params.temperature),
                            "top_p": float(params.top_p),
                            # Ensure single completion per request
                            "n": 1,
                        }
                        # CRITICAL: Use prompt_token_ids instead of decoded text
                        # This avoids the tokenize→decode→re-tokenize round-trip which can
                        # cause token mismatches if chat templates differ between miner and vLLM
                        # Must go in extra_body since OpenAI client doesn't support it natively
                        extra_body: dict[str, Any] = {
                            "prompt_token_ids": prompt_ids,  # Pass token IDs directly to vLLM
                        }
                        if params.top_k is not None:
                            extra_body["top_k"] = int(params.top_k)
                        if params.repetition_penalty is not None:
                            extra_body["repetition_penalty"] = float(params.repetition_penalty)

                        # CRITICAL: Request token IDs to avoid re-tokenization mismatch
                        # This ensures the token IDs we use match the logprobs from vLLM
                        # Note: vLLM 0.10.2+ returns both text AND token_ids when this is set
                        if self._return_chosen_logprobs:
                            extra_body["return_token_ids"] = True
                            # Ensure special tokens are preserved for exact alignment
                            extra_body["skip_special_tokens"] = False
                            extra_body["spaces_between_special_tokens"] = False
                            # Include stop string to ensure logprobs length matches tokens
                            extra_body["include_stop_str_in_output"] = True

                        if extra_body:
                            completion_kwargs["extra_body"] = extra_body
                        if rnd_seed is not None:
                            completion_kwargs["seed"] = int(rnd_seed)
                        if self._return_chosen_logprobs:
                            # Request logprobs. We only store chosen-token logprobs; top alternatives are ignored.
                            completion_kwargs["logprobs"] = 1

                        # Get client (lazy creation handles event loop changes)
                        client = self._get_client()
                        response = await client.completions.create(**completion_kwargs)
                        text = response.choices[0].text if response.choices else ""
                        chosen_logprobs: list[float] | None = None
                        chosen_token_ids: list[int] | None = None
                        try:
                            # Extract logprobs from response
                            lp = getattr(response.choices[0], "logprobs", None)
                            if lp is not None and hasattr(lp, "token_logprobs"):
                                # OpenAI-compatible field; list of floats for chosen tokens
                                chosen_logprobs = list(lp.token_logprobs or [])

                            # Extract token IDs from response (vLLM 0.10.2+)
                            # When return_token_ids=True, vLLM returns token_ids in the choice
                            choice = response.choices[0]
                            if hasattr(choice, "token_ids") and choice.token_ids is not None:
                                # Direct token_ids field (vLLM 0.10.2+)
                                chosen_token_ids = list(choice.token_ids)
                            elif lp is not None and hasattr(lp, "tokens"):
                                # Fallback: try to extract from logprobs.tokens field
                                tokens_field = lp.tokens
                                if tokens_field:
                                    try:
                                        # tokens might be integers or strings
                                        chosen_token_ids = [
                                            int(t)
                                            if isinstance(t, (int, str)) and str(t).isdigit()
                                            else None
                                            for t in tokens_field
                                        ]
                                        # If we got None values, token IDs weren't available
                                        if any(t is None for t in chosen_token_ids):
                                            chosen_token_ids = None
                                    except (ValueError, TypeError):
                                        chosen_token_ids = None
                        except Exception as e:
                            logger.debug("Failed to extract logprobs/token_ids: %s", e)
                            chosen_logprobs = None
                            chosen_token_ids = None
                        _ = time.time() - req_start
                        return (idx, text, chosen_logprobs, chosen_token_ids)
                    except RuntimeError as e:
                        # Handle event loop closure (happens during window transitions)
                        if "Event loop is closed" in str(e):
                            logger.warning(
                                "  vLLMServer req %d: Event loop closed, recreating client...",
                                idx + 1,
                            )
                            self._recreate_client()
                            # Don't count this as a retry attempt - just recreate and try again
                            if attempt < max_retries - 1:
                                await asyncio.sleep(0.5)
                                continue
                        # Other RuntimeErrors fall through to general handler
                        raise
                    except Exception as e:
                        if attempt < max_retries - 1:
                            backoff = base_backoff * (2**attempt)
                            logger.warning(
                                "  vLLMServer req %d failed (attempt %d/%d), retrying in %.1fs: %s",
                                idx + 1,
                                attempt + 1,
                                max_retries,
                                backoff,
                                type(e).__name__,
                            )
                            await asyncio.sleep(backoff)
                        else:
                            logger.warning(
                                "  vLLMServer req %d failed after %d attempts: %s",
                                idx + 1,
                                max_retries,
                                type(e).__name__,
                            )
                            return (idx, "", None, None)
            return (idx, "", None, None)

        tasks = []
        for idx, prompt_ids in enumerate(prompt_ids_batch):
            seed_val = seeds[idx] if seeds and idx < len(seeds) else None
            tasks.append(_call_one_async(idx, prompt_ids, seed_val))

        results_tuples = await asyncio.gather(*tasks, return_exceptions=False)
        completions: dict[int, str] = {}
        chosen_lp_map: dict[int, list[float] | None] = {}
        chosen_token_ids_map: dict[int, list[int] | None] = {}
        for idx, text, chosen_lp, chosen_tok_ids in results_tuples:
            completions[idx] = text
            chosen_lp_map[idx] = chosen_lp
            chosen_token_ids_map[idx] = chosen_tok_ids

        results: list[tuple[list[int], list[float] | None]] = []
        for idx, p_ids in enumerate(prompt_ids_batch):
            completion_text = completions.get(idx, "")
            vllm_token_ids = chosen_token_ids_map.get(idx)

            # CRITICAL FIX: Use actual token IDs from vLLM if available
            # This ensures token IDs match the logprobs returned by vLLM
            if vllm_token_ids is not None:
                comp_ids = vllm_token_ids
                logger.debug(
                    "Using vLLM token IDs directly (count=%d) to avoid re-tokenization mismatch",
                    len(comp_ids),
                )
            else:
                # Fallback: re-tokenize (may cause logprob mismatch)
                comp_ids = _tokenize_completion(self._tokenizer, completion_text, [])
                if self._warn_on_missing_token_ids:
                    logger.warning(
                        "vLLM did not return token IDs; falling back to re-tokenization. "
                        "This may cause importance sampling ratio mismatch!"
                    )

            all_ids = p_ids + comp_ids
            chosen_lp = chosen_lp_map.get(idx)
            results.append((all_ids, chosen_lp))

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time if batch_time > 0 else 0
        logger.info(
            "vLLMServer: ASYNC batch complete in %.2fs (%d prompts, %.1f prompts/sec)",
            batch_time,
            batch_size,
            throughput,
        )
        return results


class SGLangServerBackend:
    """sgLang server (OpenAI-compatible) backend over HTTP with async API.

    Uses AsyncOpenAI client for concurrent, non-blocking requests to a running
    SGLang server. Runs in a separate subprocess, avoiding Gloo socket corruption.

    Reference: https://docs.sglang.ai/basic_usage/openai_api.html
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        tokenizer: Any,
        timeout: float = 300.0,
        max_concurrent_requests: int = 4,
        return_chosen_logprobs: bool = False,
    ) -> None:
        from openai import AsyncOpenAI

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._timeout = float(timeout)
        self._max_concurrent_requests = max_concurrent_requests
        self._return_chosen_logprobs = bool(return_chosen_logprobs)

        # Initialize AsyncOpenAI client pointing to SGLang server
        self._client = AsyncOpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="EMPTY",  # SGLang doesn't require authentication
            timeout=self._timeout,
        )

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Generate using async OpenAI API to SGLang server.

        Now properly async - called directly from async context without nested event loops.
        """
        return await self._async_generate_batch(prompt_ids_batch, params, seeds)

    async def _async_generate_batch(
        self,
        prompt_ids_batch: list[list[int]],
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Async batch generation with concurrent requests and retry logic."""
        import asyncio

        batch_start = time.time()
        batch_size = len(prompt_ids_batch)
        logger.info("SGLangServer: Starting ASYNC batch of %d prompts", batch_size)

        prompts = _decode_prompts(self._tokenizer, prompt_ids_batch)

        # Use configurable semaphore to control client-side concurrency
        sem = asyncio.Semaphore(self._max_concurrent_requests)

        async def _call_one_async(
            idx: int, prompt: str, random_seed: int | None
        ) -> tuple[int, str]:
            """Make async OpenAI API request with retries and backoff."""
            max_retries: int = 3
            base_backoff: float = 1.0

            async with sem:  # Limit concurrency
                for attempt in range(max_retries):
                    req_start = time.time()
                    try:
                        # Build completion kwargs
                        completion_kwargs: dict[str, Any] = {
                            "model": self._model_name,
                            "prompt": prompt,
                            "max_tokens": int(params.max_new_tokens),
                            "temperature": float(params.temperature),
                            "top_p": float(params.top_p),
                            # Ensure single completion per request
                            "n": 1,
                        }
                        # Provide vendor extensions via extra_body for SGLang
                        extra_body: dict[str, Any] = {}
                        if params.top_k is not None:
                            extra_body["top_k"] = int(params.top_k)
                        if params.repetition_penalty is not None:
                            extra_body["repetition_penalty"] = float(params.repetition_penalty)

                        if extra_body:
                            completion_kwargs["extra_body"] = extra_body

                        # Add seed (SGLang supports this parameter)
                        if random_seed is not None:
                            completion_kwargs["seed"] = int(random_seed)

                        # Async call to server
                        response = await self._client.completions.create(**completion_kwargs)

                        req_time = time.time() - req_start
                        text = response.choices[0].text if response.choices else ""
                        logger.debug(
                            "  Request %d/%d took %.2fs, output_len=%d",
                            idx + 1,
                            batch_size,
                            req_time,
                            len(text),
                        )
                        return (idx, text)

                    except Exception as e:
                        req_time = time.time() - req_start
                        if attempt < max_retries - 1:
                            backoff = base_backoff * (2**attempt)
                            logger.warning(
                                "  Request %d/%d failed (attempt %d/%d), retrying in %.1fs: %s",
                                idx + 1,
                                batch_size,
                                attempt + 1,
                                max_retries,
                                backoff,
                                type(e).__name__,
                            )
                            await asyncio.sleep(backoff)
                        else:
                            logger.warning(
                                "  Request %d/%d failed after %d attempts (%.2fs): %s",
                                idx + 1,
                                batch_size,
                                max_retries,
                                req_time,
                                type(e).__name__,
                            )
                            return (idx, "")

                return (idx, "")

        # Execute all requests concurrently with gather
        tasks = []
        for idx, prompt in enumerate(prompts):
            seed_val = seeds[idx] if seeds and idx < len(seeds) else None
            tasks.append(_call_one_async(idx, prompt, seed_val))

        # Wait for all completions
        results_tuples = await asyncio.gather(*tasks, return_exceptions=False)

        # Build completion dict from results
        completions: dict[int, str] = {}
        for idx, text in results_tuples:
            completions[idx] = text

        # Reconstruct results in original order
        results: list[tuple[list[int], list[float] | None]] = []
        for idx, p_ids in enumerate(prompt_ids_batch):
            completion_text = completions.get(idx, "")
            comp_ids = _tokenize_completion(self._tokenizer, completion_text, [])
            all_ids = p_ids + comp_ids
            # SGLang backend doesn't currently support logprobs extraction
            results.append((all_ids, None))

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time if batch_time > 0 else 0
        logger.info(
            "SGLangServer: ASYNC batch complete in %.2fs (%d prompts, %.1f prompts/sec)",
            batch_time,
            batch_size,
            throughput,
        )

        return results


@dataclass
class GRPORollout:
    """Single rollout for GRPO training with GRAIL proof support."""

    tokens: list[int]
    token_logprobs: list[float]
    prompt_length: int
    completion_length: int
    reward: float
    advantage: float
    trajectory: list[tuple[Any, Any, float]]
    success: bool
    commitments: list[dict]
    signature: bytes
    beacon: dict
    proof_version: str


def _detect_model_device(model: Any, fallback: str = "cuda") -> str:
    """Detect the primary device for a model, handling multi-GPU setups.

    Args:
        model: The loaded model (may be distributed across GPUs)
        fallback: Device to use if detection fails

    Returns:
        Device string (e.g., "cuda:0" or "cuda")
    """
    # Check if model has grail_primary_device set (from multi-GPU loading)
    if hasattr(model, "grail_primary_device") and model.grail_primary_device:
        return model.grail_primary_device

    # Try to get device from model's first parameter
    try:
        first_param = next(model.parameters())
        return str(first_param.device)
    except (StopIteration, AttributeError):
        pass

    # Check model.device attribute
    if hasattr(model, "device"):
        return str(model.device)

    return fallback


class AgentEnvLoop:
    """Stateful episode driver for step-only environments.

    Handles prompt rendering, model generation with logprobs, GRAIL
    commitments, and GRPO advantage computation. Supports single and
    vectorized execution.

    Multi-GPU Support:
        When using a model distributed across multiple GPUs (device_map="auto"),
        the loop automatically detects the primary device and handles tensor
        placement correctly for proof computation.

    vLLM Backend Support:
        Set GRAIL_USE_VLLM=1 and GRAIL_VLLM_URL=http://localhost:8000 to use
        vLLM for generation (faster than HuggingFace). The HF model is still
        used for GRAIL proof computation which requires hidden states.

    Memory Optimization:
        Uses forward hooks to capture only the required hidden layer (LAYER_INDEX)
        instead of storing all 33+ layer activations. This reduces VRAM usage
        by 30-50% during proof computation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.7,
        *,
        do_sample: bool = True,
        top_p: float = 0.95,
        top_k: int | None = 50,
        repetition_penalty: float | None = 1.1,
        gen_backend: TextGenBackend | None = None,
        use_vllm: bool = False,
        vllm_url: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        # Detect device from model (handles multi-GPU)
        self.device = _detect_model_device(model, device) if model else device
        logger.info(f"AgentEnvLoop using device: {self.device}")

        # Hidden state hook for memory-efficient proof computation
        self._captured_hidden_state: torch.Tensor | None = None
        self._hidden_hook_handle: Any = None
        self._setup_hidden_hook()

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        self._gen_params = GenerationParams(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            trim_right_padding=False,
        )

        # Check environment variables for vLLM configuration
        env_use_vllm = os.getenv("GRAIL_USE_VLLM", "0") == "1"
        env_vllm_url = os.getenv("GRAIL_VLLM_URL", "http://localhost:8000")
        use_vllm = use_vllm or env_use_vllm
        vllm_url = vllm_url or env_vllm_url

        # Select generation backend
        if gen_backend is not None:
            # Explicit backend provided
            self._backend: TextGenBackend = gen_backend
            logger.info("Using provided generation backend")
        elif use_vllm:
            # Use vLLM server backend for faster generation
            # Prefer env var (set by miner after server starts), else derive from model path
            env_model_name = os.getenv("GRAIL_VLLM_MODEL_NAME")
            if env_model_name:
                model_name = env_model_name
            else:
                # Use basename only - vLLM serves as "checkpoint-XXXXX" not full path
                raw_name = getattr(model, "name_or_path", "model") if model else "model"
                model_name = Path(raw_name).name
            self._backend = VLLMServerBackend(
                base_url=vllm_url,
                model_name=model_name,
                tokenizer=tokenizer,
                return_chosen_logprobs=True,
                max_concurrent_requests=32,  # High concurrency for throughput
            )
            logger.info(f"🚀 Using vLLM backend at {vllm_url} (model={model_name})")
        else:
            # Default: HuggingFace backend
            self._backend = HFBackend(model, tokenizer, self.device)
            logger.info("Using HuggingFace backend for generation")

        # Hidden dim is always needed for GRAIL proof computation
        # (proofs use HF model regardless of generation backend)
        self._hidden_dim: int | None = None
        if model is not None:
            self._hidden_dim = resolve_hidden_size(model)

    def _setup_hidden_hook(self) -> None:
        """Setup forward hook to capture only the required hidden layer.

        MEMORY OPTIMIZATION: Instead of requesting all hidden states via
        output_hidden_states=True (which stores 33+ layer activations),
        we use a forward hook to capture only LAYER_INDEX. This saves
        30-50% VRAM during proof computation.
        """
        if self.model is None:
            return

        # Find the target layer based on model architecture
        target_layer = None
        layer_index = LAYER_INDEX  # The layer we need for GRAIL proofs

        # Try different model architectures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LlamaForCausalLM, QwenForCausalLM, etc.
            layers = self.model.model.layers
            if 0 <= layer_index < len(layers):
                target_layer = layers[layer_index]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2, GPT-J style
            layers = self.model.transformer.h
            if 0 <= layer_index < len(layers):
                target_layer = layers[layer_index]
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            # GPT-NeoX style
            layers = self.model.gpt_neox.layers
            if 0 <= layer_index < len(layers):
                target_layer = layers[layer_index]

        if target_layer is not None:
            def _capture_hidden_state(module: Any, input: Any, output: Any) -> None:
                # Capture the output of this layer (the hidden state after the layer)
                if isinstance(output, tuple):
                    # Some architectures return (hidden_state, attn_weights, ...)
                    self._captured_hidden_state = output[0].detach()
                else:
                    self._captured_hidden_state = output.detach()

            self._hidden_hook_handle = target_layer.register_forward_hook(_capture_hidden_state)
            logger.info(
                "Registered hidden state hook at layer %d for memory-efficient proofs",
                layer_index,
            )
        else:
            logger.warning(
                "Could not register hidden state hook (unknown model architecture). "
                "Falling back to output_hidden_states=True (uses more VRAM)."
            )

    def _cleanup_hidden_hook(self) -> None:
        """Remove the hidden state hook."""
        if self._hidden_hook_handle is not None:
            self._hidden_hook_handle.remove()
            self._hidden_hook_handle = None

        # Log tokenizer version information for debugging
        try:
            import tokenizers  # type: ignore
            import transformers

            logger.info(
                "MINER TOKENIZER INFO: transformers=%s, tokenizers=%s, name_or_path=%s",
                transformers.__version__,
                tokenizers.__version__,
                getattr(tokenizer, "name_or_path", "unknown"),
            )
        except Exception as e:
            logger.debug("Failed to log tokenizer version info: %s", e)

    def generate_batch_for_eval(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        *,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> list[tuple[float, bool]]:
        """Lightweight batch generation for evaluation (no GRAIL proofs/commitments).

        This is ~2x faster than run_grpo_group() as it skips expensive proof computation,
        wallet signing, and advantage calculation.

        Args:
            env_factory: Factory function to create environment instances
            count: Number of rollouts to generate
            batch_size: Number of rollouts to process per batch
            seed: Optional seed for environment reset

        Returns:
            List of (reward, success) tuples for each rollout
        """
        results: list[tuple[float, bool]] = []

        # Process in batches for efficient generation
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch_count = batch_end - batch_start

            # Create and reset batch of environments
            envs = [env_factory() for _ in range(batch_count)]
            obs_list = [env.reset(seed=seed) for env in envs]

            # Collect prompts for batch
            prompts_list = [
                [{"role": m.role, "content": m.content} for m in obs.messages] for obs in obs_list
            ]

            # Batch generate tokens (no logprobs/commitments needed for eval)
            batch_results = asyncio.run(
                self._batch_generate_tokens(prompts_list, include_logprobs=False)
            )

            # Process each rollout in the batch
            for env, (all_ids, prompt_len, _chosen_lp) in zip(envs, batch_results, strict=False):
                completion_ids = all_ids[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

                # Step environment to get reward
                _next_obs, reward, _terminated, _truncated, info = env.step(
                    ChatMessage(role="assistant", content=completion_text)
                )

                results.append((float(reward), bool(info.get("success", False))))

        return results

    def run_grpo_group(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        randomness_hex: str,
        wallet: Any,  # bt.wallet, but optional in offline mode
        *,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> list[GRPORollout]:
        """Generate multiple rollouts for GRPO with proofs and compute advantages."""
        rollouts: list[GRPORollout] = []

        # Process in batches for efficient generation
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch_count = batch_end - batch_start

            # Create and reset batch of environments
            envs = [env_factory() for _ in range(batch_count)]
            obs_list = [env.reset(seed=seed) for env in envs]

            # Collect prompts for batch
            prompts_list = [
                [{"role": m.role, "content": m.content} for m in obs.messages] for obs in obs_list
            ]

            # Batch generate tokens
            batch_results = asyncio.run(
                self._batch_generate_tokens(prompts_list, include_logprobs=False)
            )

            # Step all environments and collect rewards/info
            batch_data: list[tuple[list[int], int, float, dict]] = []
            for env, (all_ids, prompt_len, _chosen_lp) in zip(envs, batch_results, strict=False):
                completion_ids = all_ids[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

                _next_obs, reward, _terminated, _truncated, info = env.step(
                    ChatMessage(role="assistant", content=completion_text)
                )
                batch_data.append((all_ids, prompt_len, float(reward), info))

            # Batch compute commitments and logprobs (single forward pass)
            all_ids_batch = [data[0] for data in batch_data]
            prompt_lens = [data[1] for data in batch_data]

            proof_results = self._batch_compute_commitments_and_logprobs(
                all_ids_batch,
                prompt_lens,
                randomness_hex,
                wallet,
            )

            # Build rollouts from batched results
            for (all_ids, prompt_len, reward, info), (
                commitments,
                logprobs,
                signature,
                beacon,
                proof_version,
            ) in zip(batch_data, proof_results, strict=False):
                completion_ids = all_ids[prompt_len:]
                action_val = info.get("assignment", [])
                trajectory = [(0, action_val, reward)]

                rollout = GRPORollout(
                    tokens=all_ids,
                    token_logprobs=[0.0] * prompt_len + logprobs,
                    prompt_length=int(prompt_len),
                    completion_length=int(len(completion_ids)),
                    reward=reward,
                    advantage=0.0,
                    trajectory=trajectory,
                    success=bool(info.get("success", False)),
                    commitments=commitments,
                    signature=signature,
                    beacon=beacon,
                    proof_version=proof_version,
                )
                # Log prompt length in format matching validator logs for easy comparison
                logger.info(
                    "[MINER] prompt_length=%d, completion_length=%d, total_tokens=%d (seed=%s)",
                    rollout.prompt_length,
                    rollout.completion_length,
                    len(all_ids),
                    seed if seed is not None else "N/A",
                )
                rollouts.append(rollout)

        advantages = self._compute_advantages([r.reward for r in rollouts])
        for rollout, adv in zip(rollouts, advantages, strict=False):
            rollout.advantage = float(adv)

        return rollouts

    # ---------------------- Shared eval helpers ----------------------
    def render_prompt_ids_batch(self, messages_list: list[list[dict[str, str]]]) -> list[list[int]]:
        """Render a batch of chat messages to token IDs using the tokenizer's template."""
        results: list[list[int]] = []
        for messages in messages_list:
            _rendered, prompt_ids = self._render_chat(messages)
            results.append(prompt_ids)
        return results

    async def generate_from_prompt_ids_batch(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        seeds: list[int] | None = None,
        trim_right_padding: bool = False,
        include_logprobs: bool = False,
    ) -> list[tuple[list[int], int, list[float] | None]]:
        """Generate sequences for a batch of tokenized prompts.

        Args:
            prompt_ids_batch: Batch of tokenized prompts (already templated).
            seeds: Optional per-sample seeds for deterministic sampling.
            trim_right_padding: If True, trims trailing right padding from completions.
            include_logprobs: If True and supported by backend, returns chosen-token
                log probabilities (one per completion token) as the third tuple element.

        Returns:
            List of triples per sample: (all_token_ids, prompt_len, chosen_logprobs_or_none).
            - all_token_ids: Full prompt+completion token ids
            - prompt_len: Length of the prompt portion
            - chosen_logprobs_or_none: List of logprobs for chosen completion tokens, or None
              when not requested or unavailable
        """
        params = replace(self._gen_params, trim_right_padding=trim_right_padding)
        backend_results = await self._backend.generate(prompt_ids_batch, params=params, seeds=seeds)

        # Backend returns tuples of (sequence, chosen_logprobs_or_none)
        results: list[tuple[list[int], int, list[float] | None]] = []
        for (seq, chosen_lp), p_ids in zip(backend_results, prompt_ids_batch, strict=False):
            prompt_len = len(p_ids)
            # If include_logprobs is False, discard any logprobs the backend may have returned
            final_lp = chosen_lp if include_logprobs else None
            results.append((seq, prompt_len, final_lp))
        return results

    def _render_chat(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, list[int]]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        toks = self.tokenizer(rendered, return_tensors="pt", return_attention_mask=False)
        prompt_ids = toks.input_ids[0].tolist()

        # Log prompt details for debugging validation mismatches
        logger.debug(
            "MINER RENDERED PROMPT: length=%d chars, tokens=%d, first_4_tokens=%s",
            len(rendered),
            len(prompt_ids),
            prompt_ids[:4],
        )

        return rendered, prompt_ids

    async def _generate_tokens(
        self,
        prompt_ids: list[int],
        *,
        include_logprobs: bool = False,
    ) -> tuple[list[int], int, list[float] | None]:
        """Generate completion tokens; optionally return chosen-token logprobs.

        This method returns tokens trimmed of right padding. When include_logprobs
        is True and the backend supports chosen-token logprob reporting (e.g.,
        vLLM OpenAI server with logprobs enabled), the third element contains a
        list of log probabilities for the chosen completion tokens; otherwise None.
        """
        # Delegate to batch method to avoid duplication
        results = await self.generate_from_prompt_ids_batch(
            [prompt_ids],
            seeds=None,
            trim_right_padding=True,
            include_logprobs=include_logprobs,
        )
        return results[0]

    async def _batch_generate_tokens(
        self,
        prompts_list: list[list[dict[str, str]]],
        *,
        include_logprobs: bool = False,
    ) -> list[tuple[list[int], int, list[float] | None]]:
        """Batch generate completion tokens; optionally include chosen-token logprobs.

        Uses left-padding to handle variable-length prompts efficiently. When
        include_logprobs is True and supported by the backend, returns a list of
        triples (all_ids, prompt_len, chosen_logprobs_or_none) per sample.
        """
        # Render all prompts to token IDs
        prompt_ids_list: list[list[int]] = []
        for prompts in prompts_list:
            _rendered, prompt_ids = self._render_chat(prompts)
            prompt_ids_list.append(prompt_ids)

        # Delegate to generate_from_prompt_ids_batch to avoid duplication
        return await self.generate_from_prompt_ids_batch(
            prompt_ids_list,
            seeds=None,
            trim_right_padding=True,
            include_logprobs=include_logprobs,
        )

    def _trim_right_padding(
        self,
        seq: torch.Tensor,
        prompt_len: int,
    ) -> tuple[list[int], int]:
        """Trim trailing padding from sequence, preserving EOS semantics.

        Args:
            seq: Full token sequence [prompt_len + completion]
            prompt_len: Length of prompt portion

        Returns:
            (trimmed_token_ids, effective_completion_len)
        """
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        # Extract completion portion
        completion = seq[prompt_len:]

        # Case 1: pad_id != eos_id (separate tokens)
        if pad_id is not None and pad_id != eos_id:
            # Trim trailing pad_id tokens
            eff_comp = int((completion != pad_id).sum().item())
        # Case 2: pad_id == eos_id or no pad_id (same token)
        else:
            # Keep up to and including first EOS in completion
            eos_hits = (completion == eos_id).nonzero(as_tuple=False)
            if eos_hits.numel() > 0:
                eff_comp = int(eos_hits[0].item()) + 1
            else:
                eff_comp = completion.size(0)

        # Trim and convert to list
        trimmed = seq[: prompt_len + eff_comp]
        return trimmed.tolist(), eff_comp

    def _batch_compute_commitments_and_logprobs(
        self,
        all_token_ids_batch: list[list[int]],
        prompt_lens: list[int],
        randomness_hex: str,
        wallet: Any,  # bt.wallet, but optional in offline mode
    ) -> list[tuple[list[dict], list[float], bytes, dict, str]]:
        """Compute GRAIL commitments and token logprobs using unbatched forward passes.

        CRITICAL: Uses individual forward passes (no batching, no padding) to ensure
        hidden states match the validator's computation exactly. This is required for
        proof verification to succeed.

        Args:
            all_token_ids_batch: List of full token sequences (prompt + completion)
            prompt_lens: List of prompt lengths corresponding to each sequence
            randomness_hex: Hex string for randomness beacon
            wallet: Bittensor wallet for signing commitments

        Returns:
            List of tuples: (commitments, logprobs, signature, beacon, proof_version)
            one per rollout in the batch.
        """
        if self._hidden_dim is None:
            raise RuntimeError(
                "Cannot compute GRAIL proofs: hidden_dim not initialized. "
                "This likely means AgentEnvLoop was created with a "
                "server backend for evaluation only."
            )

        batch_size = len(all_token_ids_batch)
        if batch_size == 0:
            return []

        from ..protocol.grail_verifier import GRAILVerifier

        verifier = GRAILVerifier(hidden_dim=self._hidden_dim)
        r_vec = verifier.generate_r_vec(randomness_hex)

        results: list[tuple[list[dict], list[float], bytes, dict, str]] = []

        # Process each rollout individually (unbatched) to match validator
        for idx, all_token_ids in enumerate(all_token_ids_batch):
            prompt_len = prompt_lens[idx]

            # Log sequence details for first rollout
            if idx == 0:
                logger.debug(
                    "MINER UNBATCHED COMPUTATION: seq_len=%d prompt_len=%d "
                    "tokens_first_4=%s tokens_last_4=%s",
                    len(all_token_ids),
                    prompt_len,
                    all_token_ids[:4],
                    all_token_ids[-4:] if len(all_token_ids) >= 4 else all_token_ids,
                )

            # Single forward pass with no padding (matches validator exactly)
            token_tensor = torch.tensor(
                all_token_ids, dtype=torch.long, device=self.device
            ).unsqueeze(0)

            # Clear captured state before forward pass
            self._captured_hidden_state = None

            with torch.inference_mode():
                # CRITICAL: No attention_mask, no position_ids - uses model defaults
                # This ensures hidden states match validator's computation exactly
                #
                # MEMORY OPTIMIZATION: Use forward hook when available to avoid
                # storing all 33+ layer activations. Only request output_hidden_states
                # if hook is not set up (fallback for unknown architectures).
                use_hook = self._hidden_hook_handle is not None
                model_outputs = self.model(
                    token_tensor,
                    output_hidden_states=not use_hook,  # Only request if no hook
                )

                # Extract hidden states from hook or model output
                if use_hook and self._captured_hidden_state is not None:
                    # OPTIMIZED PATH: Get hidden state from hook (saves 30-50% VRAM)
                    h_layer = self._captured_hidden_state[0]  # Remove batch dim
                else:
                    # FALLBACK PATH: Get from model output (stores all layers)
                    # hidden_states shape: [1, seq_len, hidden_dim]
                    h_layer = model_outputs.hidden_states[LAYER_INDEX][0]  # Remove batch dim

                # logits shape: [1, seq_len, vocab_size]
                # Move logits to CPU (validator keeps logits on CPU as well)
                logits = model_outputs.logits[0].detach().to("cpu")

            commitments: list[dict] = []
            logprobs: list[float] = []

            # Extract commitments for this sequence
            for pos in range(len(all_token_ids)):
                commitment = verifier.create_commitment(h_layer[pos], r_vec, pos)
                commitments.append(commitment)

                # Log sample commitments for debugging
                if idx == 0 and pos in [0, prompt_len - 1, prompt_len, len(all_token_ids) - 1]:
                    logger.debug(
                        "MINER COMMITMENT pos=%d token_id=%d "
                        "sketch_hash=%s rank_hash=%s hidden_norm=%.6f",
                        pos,
                        all_token_ids[pos],
                        commitment.get("sketch_hash", "")[:16],
                        commitment.get("rank_hash", "")[:16],
                        float(h_layer[pos].norm().item()),
                    )

            # Extract logprobs for completion tokens using BATCHED log_softmax
            # For each completion token at position (prompt_len + i),
            # we need the logits from the PREVIOUS position (prompt_len - 1 + i)
            # to compute the probability of generating that token
            completion_ids = all_token_ids[prompt_len:]
            num_completion_tokens = len(completion_ids)

            if num_completion_tokens > 0:
                # Calculate start and end positions for logits slice
                logit_start = max(0, prompt_len - 1)
                logit_end = min(logits.size(0), prompt_len - 1 + num_completion_tokens)

                if logit_end > logit_start:
                    # OPTIMIZATION: Single batched log_softmax instead of per-token
                    # This is 15-25x faster for typical completion lengths (200+ tokens)
                    logits_slice = logits[logit_start:logit_end]
                    log_probs_batch = torch.log_softmax(logits_slice, dim=-1)

                    # Extract logprobs for chosen tokens
                    completion_tensor = torch.tensor(
                        completion_ids[: logit_end - logit_start], dtype=torch.long
                    )
                    # Use gather for efficient extraction of chosen token logprobs
                    chosen_logprobs = log_probs_batch.gather(
                        1, completion_tensor.unsqueeze(1)
                    ).squeeze(1)
                    logprobs = chosen_logprobs.tolist()

                    # Handle any missing positions at the start
                    if logit_start > prompt_len - 1:
                        missing_count = logit_start - (prompt_len - 1)
                        logprobs = [float("-inf")] * missing_count + logprobs
                        logger.warning(
                            "Missing logits for first %d completion tokens", missing_count
                        )

                    # Handle any missing positions at the end
                    if len(logprobs) < num_completion_tokens:
                        missing_count = num_completion_tokens - len(logprobs)
                        logprobs.extend([float("-inf")] * missing_count)
                        logger.warning(
                            "Missing logits for last %d completion tokens", missing_count
                        )
                else:
                    logger.warning(
                        "No valid logit positions for completion tokens; setting all logprobs to -inf"
                    )
                    logprobs = [float("-inf")] * num_completion_tokens

            # Sign commitments
            commitment_data = json.dumps(commitments, sort_keys=True)
            commitment_hash = hashlib.sha256(commitment_data.encode()).digest()
            if bt is None or wallet is None:
                raise RuntimeError(
                    "GRAIL proof generation requires bittensor wallet (unavailable in offline mode)"
                )
            signature = wallet.hotkey.sign(commitment_hash)

            beacon = {"randomness": randomness_hex}
            proof_version = GRAIL_PROOF_VERSION

            results.append((commitments, logprobs, signature, beacon, proof_version))

        logger.debug(
            "Completed unbatched proof computation for %d rollout(s)", len(all_token_ids_batch)
        )

        return results

    def _compute_advantages(self, rewards: list[float]) -> list[float]:
        """GRPO advantages: zero-mean within group, variance-normalized."""
        n = len(rewards)
        if n == 0:
            return []
        mean_reward = sum(rewards) / n
        centered = [r - mean_reward for r in rewards]
        std = (sum(a * a for a in centered) / n) ** 0.5
        denom = max(std, 1e-8)
        return [a / denom for a in centered]
