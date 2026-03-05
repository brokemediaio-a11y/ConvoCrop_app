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

# ─── Generation parameters (tuned for short VQA-style responses) ───
# Max tokens: keep low to match short VQA training data
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "60"))
# Follow-up turns get slightly more room
MAX_NEW_TOKENS_FOLLOWUP = int(os.getenv("MAX_NEW_TOKENS_FOLLOWUP", "80"))
# Repetition penalty: higher to combat the looping we see in eval
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.4"))
# N-gram blocking: block repeated 3-grams (tighter than 4)
GEN_NO_REPEAT_NGRAM = int(os.getenv("GEN_NO_REPEAT_NGRAM", "3"))
# Temperature: slight randomness helps break repetition loops
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
# Early stopping: stop after enough tokens form a complete sentence
MIN_TOKENS_FOR_EARLY_STOP = int(os.getenv("MIN_TOKENS_FOR_EARLY_STOP", "20"))
# Max sentences per response
MAX_SENTENCES_TURN1 = int(os.getenv("MAX_SENTENCES_TURN1", "2"))
MAX_SENTENCES_FOLLOWUP = int(os.getenv("MAX_SENTENCES_FOLLOWUP", "2"))

# ─── Context window management ───
# FastVLM 1.5B context length (in tokens)
MODEL_CONTEXT_LENGTH = int(os.getenv("MODEL_CONTEXT_LENGTH", "2048"))
# Reserve tokens for generation output
GENERATION_RESERVE = int(os.getenv("GENERATION_RESERVE", "100"))
# Image tokens (vision tower output size — typically 256-576 for FastVLM)
IMAGE_TOKEN_BUDGET = int(os.getenv("IMAGE_TOKEN_BUDGET", "576"))
# When conversation history exceeds this fraction of available budget, summarize
CONTEXT_SUMMARIZE_THRESHOLD = float(os.getenv("CONTEXT_SUMMARIZE_THRESHOLD", "0.6"))
# Max conversation turns to keep in full before compaction
MAX_FULL_HISTORY_TURNS = int(os.getenv("MAX_FULL_HISTORY_TURNS", "3"))

# ─── System prompt ───
SYSTEM_PROMPT = """You are a rice disease detection assistant. You analyze images of rice leaves to identify diseases: blast, blight, or brownspot.

Rules:
- Give short, direct answers (1-2 sentences).
- Name the disease clearly in your first response.
- For follow-up questions, answer only what was asked.
- Never repeat previous answers.
- Never ask questions back to the user.
- If you are unsure, say so honestly.
- Only discuss rice diseases. For anything else, say you can only help with rice disease detection."""

# ─── Off-topic / guardrail keywords ───
# If the model's turn-1 response doesn't mention any disease, check if the image is valid
RICE_DISEASE_KEYWORDS = ["blast", "blight", "brown spot", "brownspot", "bacterial", "lesion", "infection", "disease"]
OFF_TOPIC_RESPONSE = (
    "I'm a rice disease detection assistant and can only analyze rice leaf images. "
    "Please upload a clear photo of a rice leaf so I can help identify any diseases."
)

# ─── Server configuration ───
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Image processing
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB in bytes
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")