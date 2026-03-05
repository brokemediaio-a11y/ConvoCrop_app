"""Configuration settings for the FastAPI application."""
import os
from pathlib import Path
from typing import Literal

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model" / "checkpoints" / "best"

# Model configuration
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "apple/FastVLM-1.5B")
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR))

# Device configuration
DEVICE = os.getenv("DEVICE", "auto")  # auto, cuda, cpu
USE_HALF_PRECISION = os.getenv("USE_HALF_PRECISION", "true").lower() == "true"

# Generation parameters (matching finetuning script)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))  # Reduced for concise answers
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.3"))  # GEN_REP_PENALTY from finetuning
GEN_NO_REPEAT_NGRAM = int(os.getenv("GEN_NO_REPEAT_NGRAM", "4"))  # From finetuning script
# Early stopping: stop after complete sentence if we have enough tokens
MIN_TOKENS_FOR_EARLY_STOP = int(os.getenv("MIN_TOKENS_FOR_EARLY_STOP", "30"))

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Image processing
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB in bytes
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
