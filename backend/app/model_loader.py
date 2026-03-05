"""Model loading and initialization module."""
import os
import torch
import logging
from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from PIL import Image

from app.config import (
    BASE_MODEL_NAME,
    MODEL_PATH,
    DEVICE,
    USE_HALF_PRECISION
)

logger = logging.getLogger(__name__)

# Global model instance (singleton)
_model = None
_tokenizer = None
_image_processor = None
_vision_tower = None
_mm_projector = None
_embed_tokens = None
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
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    Load the FastVLM model with LoRA adapter and mm_projector weights.
    
    Returns:
        Tuple of (model, tokenizer, image_processor, vision_tower, mm_projector, embed_tokens, device)
    """
    global _model, _tokenizer, _image_processor, _vision_tower, _mm_projector, _embed_tokens, _device
    
    if _model is not None:
        logger.info("Model already loaded, returning existing instance")
        return _model, _tokenizer, _image_processor, _vision_tower, _mm_projector, _embed_tokens, _device
    
    device = get_device()
    
    # Determine dtype based on device and precision setting
    if device.type == "cuda" and USE_HALF_PRECISION:
        dtype = torch.float16
        logger.info("Using float16 precision for GPU")
    else:
        dtype = torch.float32
        logger.info("Using float32 precision")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")
    
    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True
    )
    base_model.to(device)
    logger.info("Base model loaded")
    
    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {MODEL_PATH}...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    logger.info("LoRA adapter loaded")
    
    # Load mm_projector weights
    proj_path = Path(MODEL_PATH) / "mm_projector.pt"
    if proj_path.exists():
        logger.info(f"Loading mm_projector weights from {proj_path}...")
        proj_state = torch.load(proj_path, map_location=device)
        restored = 0
        for name, param in model.named_parameters():
            if name in proj_state:
                param.data.copy_(proj_state[name].to(device))
                restored += 1
        logger.info(f"Restored {restored} mm_projector tensors")
    else:
        logger.warning(f"mm_projector.pt not found at {proj_path}, using pretrained weights")
    
    # Set model to eval mode
    model.eval()
    
    # Extract components for inference (must be done after PeftModel wrapping)
    inner_model = model.base_model.model
    vision_tower = inner_model.get_model().get_vision_tower()
    mm_projector = inner_model.get_model().mm_projector
    embed_tokens = inner_model.get_model().embed_tokens
    
    # Get image processor
    try:
        image_processor = getattr(vision_tower, "image_processor", None)
        if image_processor is None:
            logger.warning("Image processor not found in vision tower, creating fallback")
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor(
                size={"height": 512, "width": 512},
                crop_size={"height": 512, "width": 512},
                do_resize=True,
                do_center_crop=True,
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
    except Exception as e:
        logger.error(f"Error getting image processor: {e}")
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor(
            size={"height": 512, "width": 512},
            crop_size={"height": 512, "width": 512},
            do_resize=True,
            do_center_crop=True,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    
    logger.info("Model components extracted successfully")
    
    # Verify model is ready
    logger.info("Verifying model setup...")
    test_image = Image.new("RGB", (256, 256), (128, 128, 128))
    test_pixels = image_processor(images=test_image, return_tensors="pt")["pixel_values"]
    logger.info(f"Image processor verified: shape={test_pixels.shape}")
    del test_pixels, test_image
    torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # Store globally
    _model = model
    _tokenizer = tokenizer
    _image_processor = image_processor
    _vision_tower = vision_tower
    _mm_projector = mm_projector
    _embed_tokens = embed_tokens
    _device = device
    
    logger.info("Model loaded successfully and ready for inference")
    
    return model, tokenizer, image_processor, vision_tower, mm_projector, embed_tokens, device


def get_model() -> Tuple:
    """Get the loaded model instance, loading if necessary."""
    if _model is None:
        return load_model()
    return _model, _tokenizer, _image_processor, _vision_tower, _mm_projector, _embed_tokens, _device