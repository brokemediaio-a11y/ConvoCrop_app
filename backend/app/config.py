"""Configuration settings for the FastAPI application."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model" / "checkpoints" / "best"

# Model configuration
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "apple/FastVLM-1.5B")
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR))

# Device configuration
DEVICE = os.getenv("DEVICE", "auto")  # auto, cuda, cpu
USE_HALF_PRECISION = os.getenv("USE_HALF_PRECISION", "true").lower() == "true"

# ─── Vision token ───
IMAGE_TOKEN_INDEX = -200

# ─── Generation parameters (aligned with training config) ───
# Tight defaults keep latency under ~30s on typical GPU; CPU needs small caps too.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "72"))
MAX_NEW_TOKENS_FOLLOWUP = int(os.getenv("MAX_NEW_TOKENS_FOLLOWUP", "96"))
# Never generate more than this even if the client asks (latency guardrail).
HARD_CAP_NEW_TOKENS = int(os.getenv("HARD_CAP_NEW_TOKENS", "160"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))

# ─── Context window management ───
MAX_FULL_HISTORY_TURNS = int(os.getenv("MAX_FULL_HISTORY_TURNS", "3"))
# Raised from 640 → 900 to accommodate field context prefix (~60 tokens) appearing
# in Turn 1 + current question + compaction summary every turn. 640 was too tight
# and triggered hard compaction at Turn 4, stripping field grounding from history.
TOKEN_BUDGET_SOFT_LIMIT = int(os.getenv("TOKEN_BUDGET_SOFT_LIMIT", "900"))

# Resize longest image side before the vision tower (0 = disable). Cuts vision cost a lot.
INFERENCE_IMAGE_MAX_SIDE = int(os.getenv("INFERENCE_IMAGE_MAX_SIDE", "768"))

# PyTorch CPU threads (0 = leave PyTorch default)
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0"))

# Attention: "sdpa" uses PyTorch scaled-dot-product attention (fast on CUDA when supported).
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "sdpa")

# ─── Response quality settings ───
DEDUP_WORD_OVERLAP_THRESHOLD = float(os.getenv("DEDUP_WORD_OVERLAP_THRESHOLD", "0.75"))
QUALITY_GATE_MIN_CHARS = int(os.getenv("QUALITY_GATE_MIN_CHARS", "15"))
QUALITY_GATE_MAX_REMOVAL_RATIO = float(os.getenv("QUALITY_GATE_MAX_REMOVAL_RATIO", "0.5"))
REPHRASE_RESPONSE = (
    "I've already covered the main points about {disease}. "
    "Could you rephrase your question or ask about something more specific?"
)

# ─── System prompt ───
SYSTEM_PROMPT = """You are a rice disease detection assistant. You analyze images of rice leaves to identify diseases: blast, blight, or brownspot.

Rules:
- Be concise but complete.
- Name the disease clearly in your first response.
- CRITICAL: When a [Field Context: ...] block is present in any message, all answers about treatment, prevention, management, yield, severity, and cure must be grounded in the field-level data (incidence %, severity tier, infected area). Do not answer from the leaf image alone when field data is available.
- For follow-up questions, answer only what was asked and keep answers practical.
- For treatment, prevention, dosage, or field-management questions: use 3-5 short sentences with clear actionable steps.
- Never repeat previous answers.
- Never ask questions back to the user.
- If you are unsure, say so honestly.
- Do not invent unknown facts, exact dose numbers, or product claims. If exact dosage depends on product label or local extension guidance, say that explicitly and provide safe generic guidance.
- Only discuss rice diseases. For anything else, say you can only help with rice disease detection.
- Severity you describe from the photo refers to that leaf or patch. Field-wide estimates use the separate Field survey tool in the app, not this chat."""

# ─── Field survey CTA (shown in the chat UI only; not appended to model text) ───
FIELD_SURVEY_CTA_TEXT = (
    "I recommend using the field severity calculation option to know the exact severity of the "
    "field and I'll be able to give you better guidance."
)

# Keywords in the user question that re-show sampling steps even if guidance was already sent once
SAMPLING_REMIND_KEYWORDS = (
    "remind",
    "again",
    "repeat",
    "sampling",
    "quadrat",
    "steps",
    "how to sample",
    "field sample",
)

# ─── Off-topic / guardrail ───
# Returned when the IMAGE is off-topic (Guard 2 fires)
OFF_TOPIC_RESPONSE = (
    "I'm a rice disease detection assistant and can only analyze rice leaf images. "
    "Please upload a clear photo of a rice leaf so I can help identify any diseases."
)
# Returned when the QUESTION is off-topic (Guard 1 fires)
OFF_TOPIC_QUESTION_RESPONSE = (
    "I can only answer questions related to rice disease detection. "
    "Please ask about diagnosing, treating, or preventing rice diseases."
)

# ─── Server configuration ───
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Image processing
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB in bytes
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")