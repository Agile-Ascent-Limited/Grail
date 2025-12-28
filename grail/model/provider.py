"""Model and tokenizer provider for GRAIL.

Centralized loading functions to ensure consistent configuration across
all components (Prover, Verifier, Trainer).

Multi-GPU Support:
    - Set GRAIL_MULTI_GPU=1 to enable automatic device mapping across GPUs
    - Set GRAIL_TENSOR_PARALLEL_SIZE=N to specify number of GPUs (default: all available)
    - For 8x A100: model weights distributed across all GPUs via device_map="auto"
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Multi-GPU configuration
MULTI_GPU_ENABLED = os.getenv("GRAIL_MULTI_GPU", "0") == "1"
TENSOR_PARALLEL_SIZE = int(os.getenv("GRAIL_TENSOR_PARALLEL_SIZE", "0"))  # 0 = auto

# Quantization configuration
# Options: "none", "int8", "int4"
QUANTIZATION_MODE = os.getenv("GRAIL_QUANTIZATION", "none").lower()


def _get_quantization_config() -> dict[str, Any] | None:
    """Get quantization configuration for model loading.

    MEMORY OPTIMIZATION: Quantization reduces VRAM usage:
    - int8: ~50% reduction (uses bitsandbytes 8-bit)
    - int4: ~75% reduction (uses bitsandbytes 4-bit with NF4)

    Note: Quantization may slightly reduce inference quality but enables
    running larger models (30B+) on limited VRAM.

    Returns:
        Configuration dict for from_pretrained, or None if no quantization
    """
    if QUANTIZATION_MODE == "none":
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning(
            "Quantization requested but transformers version doesn't support BitsAndBytesConfig. "
            "Update transformers: pip install -U transformers"
        )
        return None

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        logger.warning(
            f"GRAIL_QUANTIZATION={QUANTIZATION_MODE} but bitsandbytes not installed. "
            "Install with: pip install bitsandbytes"
        )
        return None

    if QUANTIZATION_MODE == "int8":
        logger.info("🔧 Using INT8 quantization (50% VRAM reduction)")
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        }
    elif QUANTIZATION_MODE == "int4":
        logger.info("🔧 Using INT4/NF4 quantization (75% VRAM reduction)")
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
            )
        }
    else:
        logger.warning(f"Unknown quantization mode: {QUANTIZATION_MODE}, using none")
        return None


def get_tokenizer(
    model_name: str,
    *,
    chat_template: str | None = None,
) -> Any:
    """Load tokenizer with consistent configuration.

    Args:
        model_name: HuggingFace model identifier or local checkpoint path
        chat_template: Optional chat template string to install

    Returns:
        Configured AutoTokenizer instance

    Note:
        If model_name is a local directory containing chat_template.jinja,
        this file is loaded automatically since AutoTokenizer.from_pretrained()
        does not auto-load .jinja files in all transformers versions.
    """
    import transformers
    logger.info(
        "[TOKENIZER] Loading from: %s (transformers=%s)",
        model_name,
        transformers.__version__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token_id only if missing (avoid conflating pad/eos semantics)
    # Required for batching; fallback to eos_token_id if no dedicated pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.debug("Set pad_token_id to eos_token_id (model had no pad token)")

    # Auto-load chat_template.jinja if present in checkpoint directory
    # This is needed because AutoTokenizer.from_pretrained() doesn't auto-load
    # .jinja files in all transformers versions (the trainer saves templates there)
    if chat_template is None:
        model_path = Path(model_name)
        jinja_path = model_path / "chat_template.jinja"
        if model_path.is_dir() and jinja_path.is_file():
            try:
                jinja_content = jinja_path.read_text(encoding="utf-8")
                tokenizer.chat_template = jinja_content
                logger.info(
                    "[TOKENIZER] ✅ Loaded chat_template.jinja from checkpoint: %s (%d bytes)",
                    jinja_path,
                    len(jinja_content),
                )
            except Exception as e:
                logger.warning("[TOKENIZER] Failed to load chat_template.jinja: %s", e)

    # Install custom chat template if explicitly provided (overrides .jinja file)
    if chat_template is not None:
        try:
            tokenizer.chat_template = chat_template
            logger.debug("Installed custom chat template (explicit parameter)")
        except Exception as e:
            logger.warning(f"Failed to set chat template: {e}")

    # Log the actual chat_template being used for debugging prompt mismatches
    actual_template = getattr(tokenizer, "chat_template", None)
    if actual_template:
        template_hash = hash(actual_template) % 10000  # Short hash for comparison
        template_preview = actual_template[:80].replace("\n", "\\n")
        logger.info(
            "[TOKENIZER] chat_template loaded: hash=%04d, len=%d, preview='%s...'",
            template_hash,
            len(actual_template),
            template_preview,
        )
    else:
        logger.warning("[TOKENIZER] NO chat_template set - apply_chat_template may fail or use defaults")

    return tokenizer


def _get_device_map(device: str | None, tensor_parallel_size: int = 0) -> dict | str | None:
    """Compute device_map for multi-GPU model loading.

    Args:
        device: Target device or None for auto
        tensor_parallel_size: Number of GPUs to use (0 = all available)

    Returns:
        device_map configuration for from_pretrained
    """
    if not MULTI_GPU_ENABLED:
        return None

    if not torch.cuda.is_available():
        logger.warning("Multi-GPU requested but CUDA not available")
        return None

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        logger.info("Multi-GPU enabled but only 1 GPU available, using single device")
        return None

    # Determine how many GPUs to use
    target_gpus = tensor_parallel_size if tensor_parallel_size > 0 else num_gpus
    target_gpus = min(target_gpus, num_gpus)

    logger.info(f"🚀 Multi-GPU enabled: using {target_gpus}/{num_gpus} GPUs with device_map='auto'")

    # Log GPU info
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    # Use "auto" for automatic balanced distribution
    # This distributes layers across available GPUs
    return "auto"


def get_model(
    model_name: str,
    *,
    device: str | None = None,
    use_safetensors: bool = True,
    eval_mode: bool = True,
    use_flash_attention: bool = False,
    checkpoint_window: int | None = None,
    force_multi_gpu: bool = False,
) -> Any:
    """Load model with consistent configuration.

    Args:
        model_name: HuggingFace model identifier or local checkpoint path
        device: Target device ("cuda", "cpu", or None for auto-detect)
        use_safetensors: Whether to prefer safetensors format
        eval_mode: Whether to set model to eval() mode
        use_flash_attention: Whether to use Flash Attention 2 (requires flash-attn package).
                            Enabled for both training and inference when set.
        checkpoint_window: Optional checkpoint window number. If not provided, will be
                          extracted from metadata.json or parsed from the path.
        force_multi_gpu: Override GRAIL_MULTI_GPU env var to enable multi-GPU

    Returns:
        Configured model instance with preserved original name and checkpoint_window attribute
    """
    logger.debug(f"Loading model: {model_name}")

    # Check for multi-GPU mode
    use_multi_gpu = force_multi_gpu or MULTI_GPU_ENABLED
    device_map = _get_device_map(device, TENSOR_PARALLEL_SIZE) if use_multi_gpu else None

    # Auto-detect device if not specified (only used if not multi-GPU)
    if device is None and device_map is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Auto-detected device: {device}")

    # Check if this is a local checkpoint path with metadata
    original_model_name = model_name
    resolved_checkpoint_window = checkpoint_window
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                original_model_name = metadata.get("model_name", model_name)
                # Extract checkpoint_window from metadata if not explicitly provided
                if resolved_checkpoint_window is None and "window" in metadata:
                    resolved_checkpoint_window = int(metadata["window"])
                logger.debug(
                    f"Found checkpoint: {original_model_name}, window={resolved_checkpoint_window}"
                )
            except Exception as e:
                logger.debug(f"Failed to read checkpoint metadata: {e}")

        # Fallback: parse checkpoint-{window} from path if still not set
        if resolved_checkpoint_window is None and "checkpoint-" in model_name:
            try:
                checkpoint_segment = model_name.split("checkpoint-")[-1].split("/")[0]
                resolved_checkpoint_window = int(checkpoint_segment)
                logger.debug(f"Parsed checkpoint window from path: {resolved_checkpoint_window}")
            except (ValueError, IndexError):
                pass

    # Configure attention implementation
    # Enable Flash Attention for CUDA (both single and multi-GPU)
    attn_implementation = None
    use_cuda = device == "cuda" or device_map is not None

    # Check if eager attention is forced (for precision tuning)
    force_eager = os.getenv("GRAIL_FORCE_EAGER_ATTENTION", "0") == "1"

    if force_eager:
        attn_implementation = "eager"
        logger.info("🎯 Using eager attention (forced for precision)")
    else:
        # Check if flash attention should be enabled (env var or explicit parameter)
        flash_attn_env = os.getenv("GRAIL_USE_FLASH_ATTENTION", "0") == "1"
        should_use_flash = use_flash_attention or flash_attn_env

        if should_use_flash and use_cuda:
            try:
                import flash_attn  # noqa: F401

                attn_implementation = "flash_attention_2"
                logger.info("⚡ Using Flash Attention 2 for optimized inference")
            except ImportError:
                logger.warning(
                    "flash-attn not installed; falling back to default attention. "
                    "Install with: pip install flash-attn --no-build-isolation"
                )

    # Determine dtype
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32

    # Build from_pretrained kwargs
    load_kwargs: dict[str, Any] = {
        "use_safetensors": use_safetensors,
        "torch_dtype": torch_dtype,
    }

    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    # Apply quantization config if enabled
    quant_config = _get_quantization_config()
    if quant_config:
        load_kwargs.update(quant_config)
        # Quantization requires device_map to be set
        if device_map is None:
            device_map = "auto"
            logger.info("Quantization enabled, setting device_map='auto'")

    # Add device_map for multi-GPU or explicit device for single GPU
    if device_map is not None:
        load_kwargs["device_map"] = device_map
        logger.info(f"📦 Loading model with device_map='{device_map}'")
    elif device is not None:
        # For single device, we'll move after loading
        pass

    # Load model
    logger.info(f"Loading model from: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Preserve original model name for GRAIL proof validation
    model.name_or_path = original_model_name

    # Store checkpoint window for validation (avoids parsing path strings)
    model.grail_checkpoint_window = resolved_checkpoint_window

    # Move to device only if not using device_map (device_map handles placement)
    if device_map is None and device is not None:
        model = model.to(device)

    # Store the effective device for later reference
    if device_map is not None:
        # With device_map, model is distributed - use first device for reference
        model.grail_primary_device = "cuda:0"
    else:
        model.grail_primary_device = device

    # Set eval mode if requested
    if eval_mode:
        model.eval()
        logger.debug("Model set to eval mode")

    # Log model metadata
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_dtype = model.dtype
        model_config = model.config

        # Determine device info for logging
        if device_map is not None:
            device_info = f"Multi-GPU (device_map='{device_map}')"
        else:
            device_info = str(device)

        logger.info(
            f"✅ Model loaded: {original_model_name} | "
            f"Params: {total_params:,} (trainable: {trainable_params:,}) | "
            f"Dtype: {model_dtype} | Device: {device_info}"
        )
        logger.debug(
            f"Model config: vocab_size={getattr(model_config, 'vocab_size', '?')}, "
            f"hidden_size={getattr(model_config, 'hidden_size', '?')}, "
            f"num_hidden_layers={getattr(model_config, 'num_hidden_layers', '?')}, "
            f"num_attention_heads={getattr(model_config, 'num_attention_heads', '?')}"
        )
    except Exception as e:
        logger.debug(f"Failed to log model metadata: {e}")

    return model


def clear_model_and_tokenizer(model: Any | None, tokenizer: Any | None) -> tuple[None, None]:
    """Release references and aggressively reclaim GPU memory.

    Returns a pair of Nones so callers can assign:
        model, tokenizer = clear_model_and_tokenizer(model, tokenizer)
    """
    try:
        # Drop strong refs in caller by returning (None, None).
        # Local deletions here allow earlier collection if no other refs exist.
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return None, None
