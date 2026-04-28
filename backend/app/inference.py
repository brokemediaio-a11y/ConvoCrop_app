"""Inference pipeline for generating responses from the model.

Includes:
- Guard 1: Pre-inference question intent check (tiered keywords)
- Guard 2: Post-inference response validation for off-topic images
- Qwen2 chat template + model.generate(images=...) for generation
- B1: Observation prefix stripping on follow-up turns
- B2: Cross-turn sentence deduplication against previous responses
- B3: Quality gate with contextual fallback
- A1: Structured context compaction with extracted disease summary
- A2: Topic classification for compaction metadata
- C1/C2: Token budget monitoring with soft/hard limits
"""
import base64
import io
import logging
import re
import time
import torch
from typing import List, Optional, Tuple
from PIL import Image, ImageFilter, ImageStat

from app.config import (
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_FOLLOWUP,
    HARD_CAP_NEW_TOKENS,
    REPETITION_PENALTY,
    MAX_FULL_HISTORY_TURNS,
    SYSTEM_PROMPT,
    OFF_TOPIC_RESPONSE,
    OFF_TOPIC_QUESTION_RESPONSE,
    IMAGE_TOKEN_INDEX,
    USE_HALF_PRECISION,
    DEDUP_WORD_OVERLAP_THRESHOLD,
    QUALITY_GATE_MIN_CHARS,
    QUALITY_GATE_MAX_REMOVAL_RATIO,
    TOKEN_BUDGET_SOFT_LIMIT,
    REPHRASE_RESPONSE,
    SAMPLING_REMIND_KEYWORDS,
    INFERENCE_IMAGE_MAX_SIDE,
)
from app.model_loader import get_model
from app.schemas import FieldMetricsState
from app.field_metrics import classify_field_severity_band
from app.field_sampling_parse import looks_like_field_sample_submission

logger = logging.getLogger(__name__)

_DISEASE_DISPLAY_NAMES = {
    "blast": "rice blast",
    "blight": "bacterial leaf blight",
    "brownspot": "brown spot",
}

# ─── FIX B2: keywords that mark a sentence as field-grounded ───
# Sentences containing these are never removed by cross-turn deduplication.
# Field-conditioned answers repeat incidence/tier/area by design — dedup
# must not strip them or the model's response degrades to leaf-level content.
_FIELD_ANCHOR_KEYWORDS = (
    "field incidence",
    "field severity",
    "field survey",
    "incidence",
    "infected area",
    "severity index",
    "% of plants",
    "field-wide",
    "field data",
    "field context",
    "samples",
)

_IMAGE_QUALITY_REJECTION_RESPONSE = (
    "I could not assess this image reliably because photo quality is too low for diagnosis. "
    "Please re-upload a clearer rice leaf photo in good light, in focus, with the leaf filling most "
    "of the frame and taken straight-on. Avoid wet droplets, heavy shadows, and extreme angles."
)

_QUALITY_REASON_LABELS = {
    "underexposed": "too dark",
    "overexposed": "too bright",
    "blurred": "blurry",
    "low_contrast": "low contrast",
    "leaf_not_prominent": "leaf not clearly visible",
}


# ═══════════════════════════════════════════════════════════════
# Image decoding
# ═══════════════════════════════════════════════════════════════

def _resize_image_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale RGB image so the longest edge is at most max_side (faster vision encode)."""
    if max_side <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return image.resize((nw, nh), resample)


def decode_base64_image(image_string: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    try:
        if "," in image_string:
            image_string = image_string.split(",")[1]
        image_data = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# Guard 1: Pre-inference question intent check
# ═══════════════════════════════════════════════════════════════
#
# Tier 1 keywords prove the user wants disease analysis.
# Tier 2 keywords mention the domain but not specific intent.
# Off-topic signals override Tier 2 but NOT Tier 1.

_DISEASE_INTENT_KEYWORDS = [
    "disease", "infection", "lesion", "symptom", "spot",
    "blast", "blight", "brownspot", "brown spot",
    "fungus", "fungi", "bacterial", "pathogen",
    "identify", "diagnos", "classify", "detect", "analyze",
    "what do you see", "what is wrong", "what's wrong",
    "affected", "damage", "severity", "severe", "mild",
    "treatment", "treat", "cure", "manage", "spray", "control",
    "fungicide", "pesticide", "prevent", "resistant", "variety",
    "cause", "spread", "transmit",
    "yield", "harvest", "loss",
    "fertilizer", "nitrogen", "potassium", "silicon", "nutrient",
    "panicle", "collar", "sheath", "tiller", "seedling",
    "stubble", "irrigation",
    "insect", "pest", "feeding",
]

_GENERIC_CONTEXT_KEYWORDS = [
    "rice", "crop", "leaf", "plant", "field", "paddy",
    "grain", "seed", "farmer", "agriculture", "farm",
    "image", "photo", "picture",
]

_OFF_TOPIC_SIGNALS = [
    "write me", "tell me a", "compose", "create a",
    "joke", "story", "poem", "song", "sing", "dance", "music",
    "movie", "game", "sport",
    "president", "politics", "election", "celebrity",
    "who is", "who was", "who are",
    "what is the capital", "history of",
    "recipe", "cook", "bake", "eating", "taste", "delicious",
    "restaurant", "menu",
    "code", "program", "software", "app", "website",
    "phone", "laptop", "computer",
    "money", "crypto", "bitcoin", "stock market", "invest",
    "homework", "essay", "exam", "school", "university",
    "girlfriend", "boyfriend", "relationship", "dating",
    "travel", "vacation", "hotel", "flight",
    "math", "calculate", "translate", "equation",
    "workout", "fitness", "gym", "fashion", "makeup",
    "weather forecast", "temperature today",
    "news", "headline",
    "rich", "handsome", "beautiful",
]

_SHORT_QUESTION_WORD_LIMIT = 8
_OFF_TOPIC_PATTERNS = [
    re.compile(
        r"(?<!\w)" + r"\s+".join(re.escape(tok) for tok in sig.split()) + r"(?!\w)"
    )
    for sig in _OFF_TOPIC_SIGNALS
]


def _match_off_topic_signal(question_lower: str) -> Optional[str]:
    """Return the first off-topic signal matched with word-safe boundaries."""
    for sig, pat in zip(_OFF_TOPIC_SIGNALS, _OFF_TOPIC_PATTERNS):
        if pat.search(question_lower):
            return sig
    return None


def _looks_like_gibberish(question: str) -> bool:
    """
    Detect obvious keyboard-mash / nonsensical input conservatively.
    We only flag when the text has enough alphabetic content and exhibits
    very low vowel density across long tokens.
    """
    q = question.strip().lower()
    if len(q) < 6:
        return False

    tokens = re.findall(r"[a-z]+", q)
    if not tokens:
        return False

    long_tokens = [t for t in tokens if len(t) >= 6]
    if not long_tokens:
        return False

    letters = "".join(long_tokens)
    vowel_count = sum(1 for ch in letters if ch in "aeiou")
    vowel_ratio = vowel_count / max(len(letters), 1)

    # Example catches: "sfjvnskfvjkjfnv", "asdlkjqweqwe"
    return vowel_ratio < 0.18 and len(letters) >= 8


def is_question_on_topic(question: str) -> bool:
    """
    Guard 1 for turn-1 questions. Uses tiered keyword logic:
      1. Tier 1 keyword present -> always on-topic
      2. Off-topic signal present (no Tier 1) -> always off-topic
      3. Short question (<=8 words), no signals -> benefit of doubt
      4. Long question needs at least a Tier 2 keyword to pass
    """
    q_lower = question.lower().strip()
    tier1_hits = [kw for kw in _DISEASE_INTENT_KEYWORDS if kw in q_lower]
    off_topic_hit = _match_off_topic_signal(q_lower)
    generic_hits = [kw for kw in _GENERIC_CONTEXT_KEYWORDS if kw in q_lower]
    gibberish_hit = _looks_like_gibberish(question)
    words = len(q_lower.split())

    if tier1_hits:
        logger.info(
            "Guard1 turn1 allow (tier1). question='%s' tier1_hits=%s",
            question[:120],
            tier1_hits[:6],
        )
        return True

    if off_topic_hit:
        logger.info(
            "Guard1 turn1 block (off-topic). question='%s' off_topic_signal='%s'",
            question[:120],
            off_topic_hit,
        )
        return False

    if gibberish_hit and not generic_hits:
        logger.info(
            "Guard1 turn1 block (gibberish). question='%s'",
            question[:120],
        )
        return False

    if words <= _SHORT_QUESTION_WORD_LIMIT:
        logger.info(
            "Guard1 turn1 allow (short question). words=%s question='%s'",
            words,
            question[:120],
        )
        return True

    allowed = bool(generic_hits)
    logger.info(
        "Guard1 turn1 %s (generic tier2). words=%s generic_hits=%s question='%s'",
        "allow" if allowed else "block",
        words,
        generic_hits[:6],
        question[:120],
    )
    return allowed


def is_follow_up_on_topic(question: str, detected_disease: Optional[str]) -> bool:
    """
    Guard 1 for follow-up questions. More lenient because the user
    is already in a disease conversation.
      1. Tier 1 keyword -> on-topic
      2. Off-topic signal (no Tier 1) -> off-topic
      3. Active disease context -> on-topic
      4. Default -> on-topic (benefit of doubt in an active conversation)
    """
    q_lower = question.lower().strip()
    tier1_hits = [kw for kw in _DISEASE_INTENT_KEYWORDS if kw in q_lower]
    off_topic_hit = _match_off_topic_signal(q_lower)
    gibberish_hit = _looks_like_gibberish(question)

    if tier1_hits:
        logger.info(
            "Guard1 followup allow (tier1). disease=%s question='%s' tier1_hits=%s",
            detected_disease,
            question[:120],
            tier1_hits[:6],
        )
        return True

    if off_topic_hit:
        logger.info(
            "Guard1 followup block (off-topic). disease=%s question='%s' off_topic_signal='%s'",
            detected_disease,
            question[:120],
            off_topic_hit,
        )
        return False

    if gibberish_hit:
        logger.info(
            "Guard1 followup block (gibberish). disease=%s question='%s'",
            detected_disease,
            question[:120],
        )
        return False

    logger.info(
        "Guard1 followup allow (default in active chat). disease=%s question='%s'",
        detected_disease,
        question[:120],
    )
    return True


# ═══════════════════════════════════════════════════════════════
# Guard 2: Post-inference response validation (turn 1 only)
# ═══════════════════════════════════════════════════════════════

_SPECIFIC_DISEASE_NAMES = ["blast", "blight", "brown spot", "brownspot"]
_NEGATION_PREFIXES = ["no ", "not ", "without ", "free of ", "absence of "]
_GREEN_RATIO_THRESHOLD = 0.15


def _response_identifies_disease(response: str) -> bool:
    """
    Check if the model's turn-1 response positively identifies a specific
    rice disease (blast, blight, brownspot). Generic words like 'disease'
    or 'symptom' are ignored because the fine-tuned model always uses them
    -- even for non-rice images (e.g. 'no disease on the leaf').
    Negated mentions ('no blast', 'free of blight') do not count.
    """
    if not response:
        return False
    r_lower = response.lower()

    for name in _SPECIFIC_DISEASE_NAMES:
        pos = r_lower.find(name)
        if pos == -1:
            continue
        prefix = r_lower[max(0, pos - 20) : pos]
        if not any(neg in prefix for neg in _NEGATION_PREFIXES):
            return True

    return False


def _is_likely_plant_image(image: Image.Image) -> bool:
    """
    Quick pixel heuristic: plant/leaf images are dominated by green.
    Downsample to 64x64 and check what fraction of pixels have G as the
    strongest channel. Only called when no specific disease was identified
    (the ambiguous 'healthy leaf vs non-rice image' case).
    """
    small = image.resize((64, 64))
    pixels = list(small.getdata())
    green_count = sum(
        1 for r, g, b in pixels if g > r and g > b and g > 60
    )
    ratio = green_count / len(pixels)
    logger.debug(f"Green pixel ratio: {ratio:.2f} (threshold: {_GREEN_RATIO_THRESHOLD})")
    return ratio >= _GREEN_RATIO_THRESHOLD


def detect_image_quality_issue(image: Image.Image) -> Optional[str]:
    """
    Reject clearly low-quality inputs before expensive generation.
    """
    sample = image.convert("RGB").resize((224, 224))
    gray = sample.convert("L")

    pixels = list(sample.getdata())
    green_ratio = sum(1 for r, g, b in pixels if g > r and g > b and g > 60) / len(pixels)

    # IMPORTANT: let off-topic guard own clearly non-plant images.
    # This avoids quality checks (blur/dark/bright) masking off-topic detection.
    if green_ratio < _GREEN_RATIO_THRESHOLD:
        return None

    brightness = ImageStat.Stat(gray).mean[0]
    contrast = ImageStat.Stat(gray).stddev[0]
    edge_strength = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]

    if brightness < 34:
        return "underexposed"
    if brightness > 225:
        return "overexposed"
    if edge_strength < 7.2:
        return "blurred"
    if contrast < 16:
        return "low_contrast"
    return None


def get_image_quality_rejection_response() -> str:
    return _IMAGE_QUALITY_REJECTION_RESPONSE


def get_image_quality_issue_label(issue: Optional[str]) -> Optional[str]:
    if issue is None:
        return None
    return _QUALITY_REASON_LABELS.get(issue, issue)


# ═══════════════════════════════════════════════════════════════
# A2: Topic classification
# ═══════════════════════════════════════════════════════════════

_TOPIC_KEYWORDS = {
    "classification": [
        "what disease", "identify", "diagnos", "what is this", "what do you see",
    ],
    "appearance": [
        "look like", "affected area", "visual", "characteristics", "marks",
        "symptoms", "compare", "textbook",
    ],
    "spread": [
        "spread", "infect", "travel", "contagious", "rest of", "neighbouring",
        "nearby",
    ],
    "causes": [
        "cause", "why", "reason", "factor", "how does it",
    ],
    "treatment": [
        "treatment", "treat", "apply", "spray", "fungicide", "manage", "cure",
        "what can i do", "action", "steps", "practical",
    ],
    "prevention": [
        "prevent", "stop", "avoid", "protect", "control",
    ],
    "yield": [
        "yield", "loss", "harvest", "economic", "financial", "production",
        "risk", "expect",
    ],
    "nutrition": [
        "fertilizer", "nitrogen", "potassium", "silicon", "soil", "nutrient",
        "nutrition",
    ],
    "timeline": [
        "how long", "when", "time", "wait", "urgent", "quickly", "deadline",
    ],
}


def _classify_topic(question: str) -> str:
    """Classify a question into a topic category using keyword matching."""
    q_lower = question.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return topic
    return "general"


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE FIX: Strip sampling guidance from conversation history
# ═══════════════════════════════════════════════════════════════
#
# The sampling guidance text (~120 words / ~180 tokens) is appended to
# the model's response for the USER to read. It must NOT travel back
# into the model's prompt on subsequent turns — the 1.5B model doesn't
# need it and it bloats the context window, triggering expensive
# compaction cycles and dramatically increasing generation time.
#
# We strip it from history *before* building the prompt.

# Anchors for stripping — covers both old and new guidance formats.
_GUIDANCE_STRIP_MARKERS = (
    "\nTo get a picture of the whole field",   # new guidance
    "\nField-level estimate:",                  # old guidance (backward compat)
)


def _strip_guidance_from_history(
    history: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Remove the sampling guidance appendage from any assistant responses
    in conversation history before feeding to the model.

    This is the #1 performance fix: the guidance is ~180 tokens that the
    model never needs to see. Removing it keeps prompts tight and avoids
    unnecessary compaction cycles."""
    if not history:
        return history

    cleaned = []
    for q, a in history:
        # Check if this response has guidance appended (try each marker)
        for marker in _GUIDANCE_STRIP_MARKERS:
            idx = a.find(marker)
            if idx > 0:
                a = a[:idx].rstrip()
                logger.debug("Stripped sampling guidance from history turn")
                break
        cleaned.append((q, a))
    return cleaned


# ═══════════════════════════════════════════════════════════════
# A1: Structured context compaction
# ═══════════════════════════════════════════════════════════════

def _extract_classification_summary(turn1_response: str, disease: str) -> str:
    """Pull key metrics from the first classification response into a compact line."""
    parts = [f"Disease: {disease}"]

    pct = re.search(r"(\d+\.?\d*)\s*%", turn1_response)
    if pct:
        parts.append(f"Coverage: {pct.group(0)}")

    for level in ["severe", "advanced", "moderate", "mild", "early", "minor"]:
        if level in turn1_response.lower():
            parts.append(f"Severity: {level}")
            break

    count = re.search(r"(\d+)\s+(?:visible\s+)?(?:spots?|dots?|lesions?)", turn1_response.lower())
    if count:
        parts.append(f"Spot count: {count.group(1)}")

    return ". ".join(parts) + "."


def _format_context_summary(
    turn1_response: str,
    disease: str,
    field_metrics_state: Optional[FieldMetricsState] = None,
) -> str:
    """
    Compact line for context compaction: leaf classification plus optional
    field-level estimates. Field metrics are always included when present so
    they survive compaction and the model keeps responding at field level.
    """
    summary = _extract_classification_summary(turn1_response, disease)

    if field_metrics_state and field_metrics_state.last_avg_incidence_pct is not None:
        tier = field_metrics_state.field_severity_tier or "unknown"
        inc = field_metrics_state.last_avg_incidence_pct
        summary += (
            f" Field survey: {inc:.1f}% incidence, {tier} severity tier. "
            f"Respond to all management/treatment/yield questions at the field level."
        )
    return summary


def compact_conversation_history(
    conversation_history: List[Tuple[str, str]],
    disease_label: str,
    field_metrics_state: Optional[FieldMetricsState] = None,
) -> List[Tuple[str, str]]:
    """
    Replace old turns with a structured summary, keep recent turns in full.
    The summary captures disease, severity, metrics, and which topics were covered.
    """
    if len(conversation_history) <= MAX_FULL_HISTORY_TURNS:
        return conversation_history

    first_turn = conversation_history[0]
    cutoff = len(conversation_history) - MAX_FULL_HISTORY_TURNS
    old_turns = conversation_history[1:cutoff]
    recent_turns = conversation_history[cutoff:]

    if not old_turns:
        return conversation_history

    classification_summary = _format_context_summary(first_turn[1], disease_label, field_metrics_state)

    topics_covered = []
    for q, _a in old_turns:
        topic = _classify_topic(q)
        if topic not in topics_covered:
            topics_covered.append(topic)

    summary = f"{classification_summary} Topics already answered: {', '.join(topics_covered)}."
    compacted = [first_turn, ("[context summary]", summary)] + recent_turns
    logger.info(
        f"Compacted {len(old_turns)} old turns into summary "
        f"(topics: {topics_covered}), keeping {len(recent_turns)} recent turns"
    )
    return compacted


# ═══════════════════════════════════════════════════════════════
# Chat template helpers
# ═══════════════════════════════════════════════════════════════

def _build_messages(
    question: str,
    conversation_history: Optional[List[Tuple[str, str]]],
    is_followup: bool,
    field_context_prefix: Optional[str] = None,
) -> list:
    """Assemble the messages list for tokenizer.apply_chat_template().

    Field context placement — ALIGNED WITH TRAINING DATA (generate_vqa_v3.py):
    Training format: <image>\\n[Field Context: ...]\\n{question}  (Turn 1 user message)

    Strategy:
    - Turn 1 history entry: always inject field_context_prefix (matches training).
    - [context summary] history entries: also inject field_context_prefix so the
      model retains field grounding after compaction (FIX B — compaction bug).
    - Current question: always inject field_context_prefix when present.
    This means the model sees [Field Context:...] in every user turn that it was
    trained to associate with field-level answers.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        for i, (prev_q, prev_a) in enumerate(conversation_history):
            user_content = prev_q.replace("<image>", "").strip()
            if i == 0:
                # Turn 1: inject field context + prepend <image> token.
                # Matches training format where [Field Context:...] always
                # appeared in the first user message.
                if field_context_prefix and "[Field Context:" not in user_content:
                    user_content = f"{field_context_prefix}\n{user_content}"
                user_content = f"<image>\n{user_content}"
            elif (
                field_context_prefix
                and "[context summary]" in user_content
                and "[Field Context:" not in user_content
            ):
                # FIX B: compaction summary turns also get the field prefix
                # injected into the USER role, which is where the model was
                # trained to see [Field Context:...]. Without this, hard
                # compaction wipes the field signal and the model reverts to
                # leaf-level answers and tells the user to re-run the survey.
                user_content = f"{field_context_prefix}\n{user_content}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": prev_a})

    current_q = question.replace("<image>", "").strip()
    if not is_followup:
        if field_context_prefix and "[Field Context:" not in current_q:
            current_q = f"{field_context_prefix}\n{current_q}"
        current_q = f"<image>\n{current_q}"
    elif field_context_prefix:
        # Follow-up: also inject in the current question for immediate grounding.
        if "[Field Context:" not in current_q:
            current_q = f"{field_context_prefix}\n{current_q}"
    messages.append({"role": "user", "content": current_q})

    return messages


def _build_field_context_prefix(
    disease_label: Optional[str],
    field_metrics_state: Optional[FieldMetricsState],
) -> Optional[str]:
    """
    Build the exact field-context prefix expected by finetuning data.
    Returns None unless all required fields are available and valid.

    Training format (from generate_vqa_v3.py):
    [Field Context: Disease: {display_name}. Field incidence: {x}%.
     Samples: {n}. Estimated infected area: {x} ha out of {y} ha.
     Field severity index: {x}%. Field severity: {tier}.]
    """
    if field_metrics_state is None:
        return None

    # Resolve disease key: prefer explicit label, fall back to state
    disease_key = disease_label or field_metrics_state.last_detected_disease
    if disease_key not in _DISEASE_DISPLAY_NAMES:
        return None

    incidence = field_metrics_state.last_avg_incidence_pct
    samples = field_metrics_state.last_num_samples
    est_area_ha = field_metrics_state.last_estimated_infected_area_ha
    field_area_ha = field_metrics_state.last_field_area_ha
    severity_idx = field_metrics_state.last_avg_severity_pct
    tier = field_metrics_state.field_severity_tier

    if (
        incidence is None
        or samples is None
        or est_area_ha is None
        or field_area_ha is None
        or severity_idx is None
    ):
        return None

    # Recompute tier if missing or invalid, using disease-aware classification
    valid_tiers = ("low", "moderate", "high", "critical")
    if tier not in valid_tiers:
        if tier is not None:
            logger.warning(
                "Invalid field severity tier '%s'; recomputing from inc=%.1f sev=%.1f disease=%s",
                tier, incidence, severity_idx, disease_key,
            )
        tier = classify_field_severity_band(
            incidence_pct=incidence,
            severity_idx=severity_idx,
            disease=disease_key,
        )

    return (
        f"[Field Context: Disease: {_DISEASE_DISPLAY_NAMES[disease_key]}. "
        f"Field incidence: {incidence:.1f}%. "
        f"Samples: {int(samples)}. "
        f"Estimated infected area: {est_area_ha:.1f} ha out of {field_area_ha:.1f} ha. "
        f"Field severity index: {severity_idx:.1f}%. "
        f"Field severity: {tier}.]"
    )


def _tokenize_with_image_token(rendered: str, tokenizer, device) -> torch.Tensor:
    """Tokenize rendered prompt, replacing <image> with IMAGE_TOKEN_INDEX."""
    if "<image>" in rendered:
        pre, post = rendered.split("<image>", 1)
        pre_ids = tokenizer(
            pre, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        post_ids = tokenizer(
            post, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device)
    else:
        input_ids = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
    return input_ids


def _extract_assistant_response(full_decoded: str, rendered_prompt: str) -> str:
    """Extract the last assistant response from the decoded output."""
    assistant_marker = "<|im_start|>assistant\n"
    last_pos = full_decoded.rfind(assistant_marker)

    if last_pos != -1:
        response = full_decoded[last_pos + len(assistant_marker) :]
        for end_tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
            response = response.split(end_tok)[0]
        return response.strip()

    decoded_clean = re.sub(r"<\|.*?\|>", "", full_decoded).strip()
    prompt_clean = rendered_prompt.replace("<image>", "").strip()
    if prompt_clean in decoded_clean:
        return decoded_clean[
            decoded_clean.index(prompt_clean) + len(prompt_clean) :
        ].strip()

    return decoded_clean


# ═══════════════════════════════════════════════════════════════
# C1/C2: Token budget monitoring
# ═══════════════════════════════════════════════════════════════

def _check_token_budget(
    rendered: str,
    tokenizer,
    conversation_history: Optional[List[Tuple[str, str]]],
    question: str,
    is_followup: bool,
    disease_label: str,
    field_metrics_state: Optional[FieldMetricsState] = None,
    field_context_prefix: Optional[str] = None,
) -> Tuple[str, Optional[List[Tuple[str, str]]], str]:
    """
    If the rendered prompt exceeds the soft token limit, aggressively compact
    and re-render. Returns (new_rendered, updated_history, context_status).

    The field_context_prefix is passed through to _build_messages so that
    compaction summary turns receive the [Field Context:...] injection via
    the _build_messages fix, preserving field grounding across compaction.
    """
    token_count = len(tokenizer.encode(rendered, add_special_tokens=False))

    if token_count <= TOKEN_BUDGET_SOFT_LIMIT:
        return rendered, conversation_history, "ok"

    if not conversation_history or len(conversation_history) <= 1:
        logger.warning(
            f"Token count {token_count} exceeds soft limit {TOKEN_BUDGET_SOFT_LIMIT} "
            f"but history is already minimal"
        )
        return rendered, conversation_history, "near_limit"

    logger.info(
        f"Token count {token_count} exceeds soft limit {TOKEN_BUDGET_SOFT_LIMIT}, "
        f"aggressively compacting"
    )

    first_turn = conversation_history[0]
    summary = _format_context_summary(first_turn[1], disease_label, field_metrics_state)
    topics = []
    for q, _a in conversation_history:
        t = _classify_topic(q)
        if t not in topics:
            topics.append(t)
    summary += f" Topics already answered: {', '.join(topics)}."

    # The [context summary] user-turn key will receive field_context_prefix
    # injection automatically inside _build_messages (FIX B), so we do NOT
    # need to embed it manually here — just use the plain marker string.
    compacted = [first_turn, ("[context summary]", summary)]
    if len(conversation_history) > 1:
        compacted.append(conversation_history[-1])

    messages = _build_messages(question, compacted, is_followup, field_context_prefix)
    new_rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    new_count = len(tokenizer.encode(new_rendered, add_special_tokens=False))
    logger.info(f"After aggressive compaction: {new_count} tokens (was {token_count})")

    if new_count > TOKEN_BUDGET_SOFT_LIMIT:
        hard_compacted = [first_turn, ("[context summary]", summary)]
        messages = _build_messages(question, hard_compacted, is_followup, field_context_prefix)
        new_rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        logger.info("Hard compaction: dropped all history except summary")
        return new_rendered, hard_compacted, "compacted"

    return new_rendered, compacted, "compacted"


# ═══════════════════════════════════════════════════════════════
# B1: Observation prefix stripping
# ═══════════════════════════════════════════════════════════════

_FOLLOWUP_PREFIX_RE = re.compile(
    r"^(?:Looking at|Examining|Observing|From)\s+"
    r"(?:the\s+)?(?:uploaded\s+)?(?:leaf|image|photo|picture)"
    r"(?:\s+(?:you\s+)?(?:shared|uploaded|provided))?\s*[,;:]\s*",
    re.IGNORECASE,
)
_FOLLOWUP_BASED_ON_IMAGE_RE = re.compile(
    r"^Based on\s+(?:the\s+)?(?:uploaded\s+)?(?:leaf|image|photo|picture)"
    r"(?:\s+(?:you\s+)?(?:shared|uploaded|provided))?\s*[,;:]\s*",
    re.IGNORECASE | re.DOTALL,
)

_LONG_FIELD_PREAMBLE_RE = re.compile(
    r"^Based on the field survey showing .*?(?:in this image,\s*|,\s*)",
    re.IGNORECASE | re.DOTALL,
)
_LONG_FIELD_PREAMBLE_MARKER = "based on the field survey showing"


def _shorten_repeated_field_preamble(
    response: str,
    previous_responses: List[str],
) -> str:
    """
    Keep the detailed field-survey preamble once, then shorten repeats.

    Behavior:
    - First field-grounded answer can keep the full lead-in.
    - If a previous assistant response already used that long preamble,
      any new response starting with it is rewritten to:
      "Based on field severity, ..."
    """
    if not response or not previous_responses:
        return response

    if not any(_LONG_FIELD_PREAMBLE_MARKER in (p or "").lower() for p in previous_responses):
        return response

    if not _LONG_FIELD_PREAMBLE_RE.match(response):
        return response

    shortened = _LONG_FIELD_PREAMBLE_RE.sub("Based on field severity, ", response, count=1).strip()
    if shortened != response:
        logger.info("Shortened repeated field preamble for token efficiency")
    return shortened


def _strip_observation_prefix(response: str) -> str:
    """
    Remove the opening observation clause that re-describes the classification.
    E.g. 'Looking at the 10 round brown dots covering 2.6% of this leaf, ...'
    """
    match = _FOLLOWUP_PREFIX_RE.match(response) or _FOLLOWUP_BASED_ON_IMAGE_RE.match(response)
    if match:
        stripped = response[match.end():]
        if len(stripped) >= QUALITY_GATE_MIN_CHARS:
            logger.info(f"Stripped observation prefix: '{match.group()[:60]}...'")
            return stripped[0].upper() + stripped[1:] if stripped else stripped
    return response


# ═══════════════════════════════════════════════════════════════
# B2: Cross-turn sentence deduplication
# ═══════════════════════════════════════════════════════════════

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _PUNCT_RE.sub("", text.lower()).strip()


def _word_overlap_ratio(a: str, b: str) -> float:
    """Fraction of shared words relative to the shorter sentence."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def _deduplicate_against_history(
    response: str,
    previous_responses: List[str],
) -> Tuple[str, int, int]:
    """
    Remove sentences from response that duplicate content in previous_responses.
    Returns (cleaned_response, total_sentence_count, removed_count).

    FIX B2: sentences containing field-metric keywords (_FIELD_ANCHOR_KEYWORDS)
    are always kept regardless of overlap. Field-conditioned VQA answers repeat
    incidence/tier/area by design; dedup must not strip that grounding or the
    model falls back to leaf-level content.
    """
    if not previous_responses:
        return response, 0, 0

    history_sentences_norm = set()
    history_sentences_raw = []
    for prev in previous_responses:
        for s in re.split(r"(?<=[.!?])\s+", prev):
            s = s.strip()
            if s:
                history_sentences_norm.add(_normalize(s))
                history_sentences_raw.append(s)

    sentences = re.split(r"(?<=[.!?])\s+", response)
    kept = []
    removed = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Never deduplicate field-grounded sentences — they reference real
        # field metrics that the user calculated and the model must honour.
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in _FIELD_ANCHOR_KEYWORDS):
            kept.append(sent)
            continue

        norm = _normalize(sent)

        if norm in history_sentences_norm:
            logger.debug(f"Exact dedup: '{sent[:50]}...'")
            removed += 1
            continue

        is_fuzzy_dup = False
        for hist_sent in history_sentences_raw:
            if _word_overlap_ratio(norm, _normalize(hist_sent)) >= DEDUP_WORD_OVERLAP_THRESHOLD:
                logger.debug(f"Fuzzy dedup: '{sent[:50]}...'")
                is_fuzzy_dup = True
                break

        if is_fuzzy_dup:
            removed += 1
        else:
            kept.append(sent)

    total = len(sentences)
    return " ".join(kept).strip(), total, removed


# ═══════════════════════════════════════════════════════════════
# B3: Quality gate
# ═══════════════════════════════════════════════════════════════

def _apply_quality_gate(
    cleaned: str,
    total_sentences: int,
    removed_count: int,
    detected_disease: Optional[str],
) -> str:
    """
    If too much content was removed by dedup, return a rephrase-request fallback.
    """
    if total_sentences == 0:
        return cleaned

    removal_ratio = removed_count / total_sentences if total_sentences else 0.0

    if len(cleaned) < QUALITY_GATE_MIN_CHARS or removal_ratio > QUALITY_GATE_MAX_REMOVAL_RATIO:
        disease_name = detected_disease or "this disease"
        fallback = REPHRASE_RESPONSE.format(disease=disease_name)
        logger.info(
            f"Quality gate triggered: {removed_count}/{total_sentences} sentences removed "
            f"({removal_ratio:.0%}), response len={len(cleaned)}"
        )
        return fallback

    return cleaned


# ═══════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════

_ALL_DISEASE_NAMES = {
    "blast": ["rice blast", "blast"],
    "blight": ["bacterial blight", "blight"],
    "brownspot": ["brown spot", "brownspot", "brown_spot"],
}


def _remove_cross_disease_contamination(
    text: str, primary_disease: Optional[str] = None
) -> str:
    """Remove sentences that mention a different disease than the primary one."""
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return text

    detected = primary_disease
    if not detected:
        first_lower = sentences[0].lower()
        for disease, keywords in _ALL_DISEASE_NAMES.items():
            if any(kw in first_lower for kw in keywords):
                detected = disease
                break

    if not detected:
        return text

    other_keywords = []
    for disease, keywords in _ALL_DISEASE_NAMES.items():
        if disease != detected:
            other_keywords.extend(keywords)

    clean = [sentences[0]]
    for sent in sentences[1:]:
        if any(kw in sent.lower() for kw in other_keywords):
            logger.info(f"Removed cross-disease contamination: '{sent[:60]}...'")
            continue
        clean.append(sent)

    return " ".join(clean)


def _clean_response(text: str, detected_disease: Optional[str] = None) -> str:
    """Clean and validate model output (within-response dedup + sentence boundary)."""
    if not text or not text.strip():
        return ""

    text = text.strip()
    text = _remove_cross_disease_contamination(text, detected_disease)

    if text and text[-1] not in ".!?":
        last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_end > 0:
            text = text[: last_end + 1]
        else:
            text = text.rstrip(",;:— -") + "."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    deduped = []
    for s in sentences:
        s_norm = s.strip().lower()
        if s_norm and s_norm not in seen:
            seen.add(s_norm)
            deduped.append(s.strip())
    text = " ".join(deduped)

    return text.strip()


# ═══════════════════════════════════════════════════════════════
# Field sampling guidance (appended in API layer after generation)
# ═══════════════════════════════════════════════════════════════


def _question_triggers_sampling_guidance(question: str) -> bool:
    """Severity, prevention, treatment, yield, or field-scale topics."""
    q = question.lower()
    triggers = (
        "severity",
        "severe",
        "prevention",
        "prevent",
        "preventive",
        "cure",
        "treat",
        "treatment",
        "remedy",
        "measure",
        "measures",
        "stop the spread",
        "spread of",
        "spread this",
        "fungicide",
        "spray",
        "control",
        "manage",
        "yield",
        "yield loss",
        "harvest loss",
        "crop loss",
        "paddy",
        "whole field",
        "entire field",
        "field-wide",
        "field level",
        "across the field",
        "acres",
        "hectare",
        "hectares",
        "kanal",
        "marla",
    )
    return any(t in q for t in triggers)


def _user_wants_sampling_reminder(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in SAMPLING_REMIND_KEYWORDS)


def maybe_append_sampling_guidance(
    question: str,
    response: str,
    state: Optional[FieldMetricsState],
) -> Tuple[str, bool, FieldMetricsState, bool]:
    """
    When the topic matches (yield, prevention, etc.), signal the client to show a
    one-line field-survey CTA with a button. The model response text is unchanged;
    field metrics in state mean we never add another CTA.
    """
    st = state or FieldMetricsState()
    if response in (OFF_TOPIC_RESPONSE, OFF_TOPIC_QUESTION_RESPONSE):
        return response, False, st, False
    if looks_like_field_sample_submission(question):
        return response, False, st, False
    if st.last_avg_incidence_pct is not None:
        return response, False, st, False
    remind = _user_wants_sampling_reminder(question)
    if st.sampling_guidance_offered and not remind:
        return response, False, st, False
    if not _question_triggers_sampling_guidance(question) and not remind:
        return response, False, st, False
    new_st = st.model_copy(update={"sampling_guidance_offered": True})
    return response, True, new_st, True


# ═══════════════════════════════════════════════════════════════
# Main generation function
# ═══════════════════════════════════════════════════════════════

def generate_answer(
    image: Image.Image,
    question: str,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
    max_new_tokens: Optional[int] = None,
    detected_disease: Optional[str] = None,
    field_metrics_state: Optional[FieldMetricsState] = None,
) -> Tuple[str, str]:
    """
    Generate answer using the Qwen2 chat template and model.generate().

    Post-processing pipeline:
        B1  strip observation prefix
        B2  cross-turn sentence dedup (field-grounded sentences exempt)
        B3  quality gate (fallback if too much was removed)
        +   within-response cleanup (sentence boundary, self-dedup, cross-disease)

    Returns (response_text, context_status).
    """
    model, tokenizer, image_processor, device = get_model()

    is_followup = bool(conversation_history)
    context_status = "ok"
    logger.info(
        "GEN_START | is_followup=%s | question='%s' | history_turns=%s | detected_disease=%s | "
        "has_field_state=%s",
        is_followup,
        question[:180],
        len(conversation_history) if conversation_history else 0,
        detected_disease,
        field_metrics_state is not None,
    )

    # ── Guard 1: Pre-inference question intent check ──
    if is_followup:
        if not is_follow_up_on_topic(question, detected_disease):
            logger.info(f"Guard 1 blocked follow-up: '{question[:60]}'")
            return OFF_TOPIC_QUESTION_RESPONSE, "ok"
    else:
        if not is_question_on_topic(question):
            logger.info(f"Guard 1 blocked question: '{question[:60]}'")
            return OFF_TOPIC_QUESTION_RESPONSE, "ok"

    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS_FOLLOWUP if is_followup else MAX_NEW_TOKENS
    max_new_tokens = int(max_new_tokens)

    try:
        t0 = time.time()

        # ── Collect previous assistant responses BEFORE compaction (for B2) ──
        previous_responses: List[str] = []
        if conversation_history:
            previous_responses = [a for _q, a in conversation_history]

        # ── PERF FIX: Strip sampling guidance from history ──
        # The guidance text is for the user, not the model. Removing it
        # saves ~180 tokens per turn, preventing budget overflows and
        # expensive compaction cycles.
        if conversation_history:
            conversation_history = _strip_guidance_from_history(conversation_history)

        t1 = time.time()

        # ── Process image (optional downscale for latency) ──
        if INFERENCE_IMAGE_MAX_SIDE > 0:
            image = _resize_image_max_side(image, INFERENCE_IMAGE_MAX_SIDE)
        dtype = (
            torch.float16
            if device.type == "cuda" and USE_HALF_PRECISION
            else torch.float32
        )
        pixel_values = image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device, dtype=dtype)

        t2 = time.time()

        # ── A1: Compact conversation history ──
        if is_followup and conversation_history:
            disease_label = detected_disease or "unknown"
            conversation_history = compact_conversation_history(
                conversation_history, disease_label, field_metrics_state
            )

        field_context_prefix = _build_field_context_prefix(detected_disease, field_metrics_state)
        if field_context_prefix:
            logger.info("FIELD_CONTEXT_PREFIX_BUILT | %s", field_context_prefix)
        else:
            logger.info("FIELD_CONTEXT_PREFIX_BUILT | None")

        # When field context is present, allow a longer answer budget.
        # Applies to ALL turns (not just follow-ups) since field-conditioned
        # answers need the grounding phrase + advice to fit within the limit.
        if field_context_prefix:
            max_new_tokens = max(max_new_tokens, 112)
            logger.info(
                "Token budget uplift for field-context turn: %s -> %s",
                max_new_tokens,
                max_new_tokens,
            )
        max_new_tokens = min(max_new_tokens, HARD_CAP_NEW_TOKENS)

        # ── Build prompt ──
        messages = _build_messages(
            question, conversation_history, is_followup, field_context_prefix
        )
        first_user_turn = next(
            (m["content"] for m in messages if m.get("role") == "user"),
            "",
        )
        last_user_turn = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        logger.info(
            "PROMPT_BUILD | messages=%s | first_user_turn='%s'",
            len(messages),
            first_user_turn[:500],
        )
        if is_followup:
            logger.info(
                "PROMPT_BUILD | last_user_turn='%s'",
                last_user_turn[:300],
            )
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        logger.info("PROMPT_RENDERED_PREVIEW | %s", rendered[:600])

        # ── C1/C2: Token budget check (may re-compact) ──
        if is_followup and conversation_history:
            rendered, conversation_history, budget_status = _check_token_budget(
                rendered, tokenizer, conversation_history, question,
                is_followup, detected_disease or "unknown", field_metrics_state, field_context_prefix,
            )
            if budget_status != "ok":
                context_status = budget_status

        t3 = time.time()

        # ── Tokenize + generate ──
        input_ids = _tokenize_with_image_token(rendered, tokenizer, device)
        attention_mask = torch.ones_like(input_ids).to(device)

        t4 = time.time()
        prompt_tokens = input_ids.shape[1]
        logger.info(
            f"⏱ TIMING | history+strip={t1-t0:.2f}s | "
            f"image_proc={t2-t1:.2f}s | "
            f"compact+template+budget={t3-t2:.2f}s | "
            f"tokenize={t4-t3:.2f}s | "
            f"prompt_tokens={prompt_tokens} | "
            f"max_new_tokens={max_new_tokens}"
        )

        generate_kwargs = dict(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output_ids = model.generate(**generate_kwargs)
            else:
                output_ids = model.generate(**generate_kwargs)

        t5 = time.time()
        if output_ids.shape[1] >= prompt_tokens:
            gen_tokens = output_ids.shape[1] - prompt_tokens
        else:
            # Some model wrappers return only generated ids.
            gen_tokens = output_ids.shape[1]
        logger.info(
            f"⏱ TIMING | model.generate={t5-t4:.2f}s | "
            f"generated_tokens={gen_tokens} | "
            f"tokens_per_sec={gen_tokens/(t5-t4):.1f}"
        )

        # ── Extract raw response ──
        full_decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        response = _extract_assistant_response(full_decoded, rendered)
        raw_response = response
        logger.info("RAW_RESPONSE | len=%s | text='%s'", len(raw_response), raw_response[:400])

        # ── Within-response cleanup (sentence boundary, self-dedup, cross-disease) ──
        response = _clean_response(response, detected_disease)
        logger.info("AFTER_CLEAN_RESPONSE | len=%s | text='%s'", len(response), response[:400])

        # ── Guard 2: Post-inference response validation (turn 1 only) ──
        # Step A: model identified a specific disease -> valid, skip check.
        # Step B: no disease identified -> check if image looks like a plant.
        #         If green content is too low, the image is likely non-rice.
        if not is_followup and not _response_identifies_disease(response):
            if not _is_likely_plant_image(image):
                logger.info(
                    f"Guard 2 blocked: no disease identified and image lacks "
                    f"plant content. Response preview: '{response[:80]}'"
                )
                del pixel_values, input_ids, attention_mask, output_ids
                return OFF_TOPIC_RESPONSE, "ok"
            logger.info(
                "No disease identified but image appears plant-like — "
                "allowing healthy-leaf response through"
            )

        # ── B1: Strip observation prefix on follow-ups ──
        if is_followup and response:
            before_strip = response
            response = _strip_observation_prefix(response)
            if response != before_strip:
                logger.info(
                    "AFTER_PREFIX_STRIP | before_len=%s | after_len=%s | text='%s'",
                    len(before_strip),
                    len(response),
                    response[:400],
                )
            else:
                logger.info("AFTER_PREFIX_STRIP | unchanged | len=%s", len(response))

        # ── Token efficiency: shorten repeated field preamble ──
        # Keep the first detailed "Based on the field survey showing ..."
        # occurrence, then collapse repeats in later turns.
        if is_followup and response and previous_responses:
            before_short = response
            response = _shorten_repeated_field_preamble(response, previous_responses)
            if response != before_short:
                logger.info(
                    "AFTER_FIELD_PREAMBLE_SHORTEN | before_len=%s | after_len=%s | text='%s'",
                    len(before_short),
                    len(response),
                    response[:400],
                )

        # ── B2: Cross-turn deduplication ──
        if is_followup and response and previous_responses:
            response, total_sents, removed_sents = _deduplicate_against_history(
                response, previous_responses
            )
            if removed_sents > 0:
                logger.info(
                    f"Cross-turn dedup: removed {removed_sents}/{total_sents} sentences"
                )

            # ── B3: Quality gate ──
            before_qg = response
            response = _apply_quality_gate(
                response, total_sents, removed_sents, detected_disease
            )
            if response != before_qg:
                logger.info(
                    "AFTER_QUALITY_GATE | changed | before_len=%s | after_len=%s | text='%s'",
                    len(before_qg),
                    len(response),
                    response[:400],
                )
            else:
                logger.info("AFTER_QUALITY_GATE | unchanged | len=%s", len(response))

        if not is_followup and response:
            logger.info(f"Turn-1 response: {response[:120]}")
        logger.info("FINAL_RESPONSE | len=%s | text='%s'", len(response), response[:500])

        # ── Release references (avoid per-request empty_cache; it synchronizes the GPU) ──
        del pixel_values, input_ids, attention_mask, output_ids

        t6 = time.time()
        logger.info(
            f"⏱ TIMING TOTAL | total={t6-t0:.2f}s | "
            f"pre_generate={t4-t0:.2f}s | generate={t5-t4:.2f}s | "
            f"post_process={t6-t5:.2f}s"
        )

        return response, context_status

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise


# ═══════════════════════════════════════════════════════════════
# Disease extraction
# ═══════════════════════════════════════════════════════════════

def infer_disease(text: str) -> str:
    """Extract disease from model response using keyword scoring."""
    if not text or not text.strip():
        return "unknown"

    response_lower = text.lower()
    pred_disease = "unknown"
    max_score = 0

    disease_keywords = {
        "blast": [
            "blast", "magnaporthe", "pyricularia",
            "spindle-shaped", "diamond-shaped",
        ],
        "brownspot": [
            "brown spot", "brownspot", "brown_spot",
            "sesame-seed", "cochliobolus",
        ],
        "blight": [
            "blight", "bacterial", "xanthomonas", "water-soaked",
        ],
    }

    for disease, keywords in disease_keywords.items():
        score = sum(1 for kw in keywords if kw in response_lower)
        if score > max_score:
            max_score = score
            pred_disease = disease

    return pred_disease