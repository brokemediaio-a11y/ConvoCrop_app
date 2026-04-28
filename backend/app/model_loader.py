"""Model loading and initialization module.

Aligned with the evaluation script's loading strategy:
- Uses device_map for placement instead of manual .to()
- Relies on model.generate(images=...) for vision processing
- Extracts only the image_processor from the vision tower

PERF FIX: merge LoRA weights into base model at load time.
This eliminates CPU-GPU tensor transfers during generation that
were causing 140s+ inference times on a 4050 Ti.
"""
import torch
import logging
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.config import (
    BASE_MODEL_NAME,
    MODEL_PATH,
    DEVICE,
    USE_HALF_PRECISION,
    ATTN_IMPLEMENTATION,
    TORCH_NUM_THREADS,
)

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_image_processor = None
_device = None


def get_device() -> torch.device:
    """Determine and return the appropriate device."""
    global _device
    if _device is not None:
        return _device

    if DEVICE == "auto":
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            _device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    elif DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    return _device


def load_model() -> Tuple:
    """
    Load the FastVLM base model + LoRA adapter.

    Returns (model, tokenizer, image_processor, device).
    """
    global _model, _tokenizer, _image_processor, _device

    if _model is not None:
        logger.info("Model already loaded, returning existing instance")
        return _model, _tokenizer, _image_processor, _device

    device = get_device()

    if device.type == "cpu" and TORCH_NUM_THREADS > 0:
        torch.set_num_threads(TORCH_NUM_THREADS)
        logger.info(f"torch.set_num_threads({TORCH_NUM_THREADS})")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    dtype = (
        torch.float16
        if device.type == "cuda" and USE_HALF_PRECISION
        else torch.float32
    )
    logger.info(f"Using {dtype} precision on {device}")

    logger.info(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
    except Exception as e:
        logger.warning(
            f"Adapter tokenizer failed ({e}), falling back to base model tokenizer"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")

    logger.info(f"Loading base model: {BASE_MODEL_NAME}...")
    load_kw: dict = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=str(device),
    )
    if device.type == "cuda" and ATTN_IMPLEMENTATION:
        load_kw["attn_implementation"] = ATTN_IMPLEMENTATION
    try:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **load_kw)
    except (TypeError, ValueError, OSError) as e:
        load_kw.pop("attn_implementation", None)
        logger.warning(
            "Could not load with attn_implementation=%s (%s); retrying without it",
            ATTN_IMPLEMENTATION,
            e,
        )
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **load_kw)
    logger.info("Base model loaded")

    # ── Extract image processor BEFORE merging ──
    # The vision tower is on the base model and stays intact through merge.
    vt = (
        base_model.get_vision_tower()
        if hasattr(base_model, "get_vision_tower")
        else base_model.model.get_vision_tower()
    )
    image_processor = vt.image_processor
    logger.info("Image processor extracted from vision tower")

    # ── Load LoRA adapter ──
    logger.info(f"Loading LoRA adapter from {MODEL_PATH}...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    # ── PERF FIX: Merge LoRA weights into the base model ──
    #
    # PeftModel.from_pretrained() loads adapter layers to CPU by default.
    # During generate(), every forward pass transfers tensors between
    # CPU (adapter) and GPU (base model), adding ~2s per generated token.
    #
    # merge_and_unload() folds the LoRA deltas (A*B matrices) directly
    # into the base model weight tensors on GPU. After merging:
    #   - No more CPU<->GPU transfers during generation
    #   - Slightly less VRAM (no separate adapter tensors)
    #   - Same output quality (mathematically identical)
    #   - Cannot resume training from this merged model (fine for inference)
    logger.info("Merging LoRA weights into base model for fast inference...")
    model = model.merge_and_unload()
    logger.info("LoRA merge complete - all weights on single device")

    model.eval()

    _model = model
    _tokenizer = tokenizer
    _image_processor = image_processor
    _device = device

    # Log VRAM usage after loading
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"VRAM after load: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    logger.info("Model loaded successfully and ready for inference")
    return model, tokenizer, image_processor, device


def get_model() -> Tuple:
    """Get the loaded model instance, loading if necessary."""
    if _model is None:
        return load_model()
    return _model, _tokenizer, _image_processor, _device