"""Inference pipeline for generating responses from the model."""
import base64
import io
import logging
import re
import torch
from typing import List, Optional, Tuple
from PIL import Image

from app.config import (
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_FOLLOWUP,
    REPETITION_PENALTY,
    GEN_NO_REPEAT_NGRAM,
    MIN_TOKENS_FOR_EARLY_STOP,
    TEMPERATURE,
    MAX_SENTENCES_TURN1,
    MAX_SENTENCES_FOLLOWUP,
    MODEL_CONTEXT_LENGTH,
    GENERATION_RESERVE,
    IMAGE_TOKEN_BUDGET,
    CONTEXT_SUMMARIZE_THRESHOLD,
    MAX_FULL_HISTORY_TURNS,
    SYSTEM_PROMPT,
    RICE_DISEASE_KEYWORDS,
    OFF_TOPIC_RESPONSE,
)
from app.model_loader import get_model

logger = logging.getLogger(__name__)

IMAGE_TOKEN_INDEX = -200


# ═══════════════════════════════════════════════════════════════
# Image decoding
# ═══════════════════════════════════════════════════════════════

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
# Off-topic / guardrail detection
# ═══════════════════════════════════════════════════════════════

# Questions that are ON-topic (rice/crop/agriculture related)
_ON_TOPIC_KEYWORDS = [
    # Disease names
    "blast", "blight", "brown spot", "brownspot", "disease", "infection",
    "lesion", "spot", "symptom", "fungus", "fungi", "bacterial", "pathogen",
    # Crop / agriculture terms
    "rice", "crop", "leaf", "plant", "field", "paddy", "grain", "seed",
    "harvest", "yield", "soil", "fertilizer", "nitrogen", "silicon",
    "fungicide", "pesticide", "spray", "irrigation", "water", "stubble",
    "panicle", "collar", "sheath", "tiller", "seedling", "planting",
    "variety", "resistant", "extension", "farmer", "agriculture",
    # Common follow-up patterns about the diagnosed disease
    "prevent", "cause", "spread", "treatment", "cure", "manage", "control",
    "risk", "damage", "affect", "action", "wait", "save", "protect",
    "insect", "feeding", "cycle", "adapt",
    # Image analysis
    "image", "photo", "picture", "see", "look", "show", "diagnos",
]

# Questions that are clearly OFF-topic
_OFF_TOPIC_PATTERNS = [
    "weather", "recipe", "cook", "code", "program", "software",
    "math", "calculate", "translate", "poem", "story", "joke", "song",
    "who is", "what is the capital", "movie", "game", "sport", "news",
    "rich", "handsome", "beautiful", "money", "crypto", "bitcoin",
    "girlfriend", "boyfriend", "date", "love", "relationship",
    "write me", "tell me a", "sing", "dance", "music",
    "car", "house", "phone", "laptop", "computer",
    "homework", "essay", "exam", "school", "university",
]


def is_question_on_topic(question: str) -> bool:
    """
    Check if a user question is related to rice/crop disease.
    Returns True if on-topic.
    
    Strategy: off-topic patterns override unless on-topic keywords are present.
    For ambiguous questions like 'what do you see' paired with an image, default on-topic.
    """
    q_lower = question.lower().strip()

    has_off_topic = any(pat in q_lower for pat in _OFF_TOPIC_PATTERNS)
    has_on_topic = any(kw in q_lower for kw in _ON_TOPIC_KEYWORDS)

    if has_on_topic:
        return True
    if has_off_topic:
        return False

    # Ambiguous — generic questions like "what do you see" are on-topic for turn 1
    return True


def is_follow_up_on_topic(question: str, detected_disease: Optional[str]) -> bool:
    """
    Check if a follow-up question is on-topic given conversation context.
    More lenient since user is already in a disease conversation.
    """
    q_lower = question.lower().strip()

    has_off_topic = any(pat in q_lower for pat in _OFF_TOPIC_PATTERNS)
    has_on_topic = any(kw in q_lower for kw in _ON_TOPIC_KEYWORDS)

    if has_off_topic and not has_on_topic:
        return False
    if detected_disease:
        return True
    return has_on_topic


# ═══════════════════════════════════════════════════════════════
# Context window management
# ═══════════════════════════════════════════════════════════════

def estimate_token_count(text: str, tokenizer) -> int:
    """Fast token count estimation for context budget."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def compact_conversation_history(
    conversation_history: List[Tuple[str, str]],
    disease_label: str,
    tokenizer,
) -> List[Tuple[str, str]]:
    """
    Compact older conversation turns into a summary to free context window.
    Keeps the most recent MAX_FULL_HISTORY_TURNS in full, summarizes the rest.
    """
    if len(conversation_history) <= MAX_FULL_HISTORY_TURNS:
        return conversation_history

    cutoff = len(conversation_history) - MAX_FULL_HISTORY_TURNS
    old_turns = conversation_history[:cutoff]
    recent_turns = conversation_history[cutoff:]

    topics_covered = []
    for q, a in old_turns:
        q_short = " ".join(q.split()[:8]).rstrip("?.!,")
        topics_covered.append(q_short)

    summary = f"Disease: {disease_label}. Previously discussed: {'; '.join(topics_covered)}."
    compacted = [("[context summary]", summary)] + recent_turns
    logger.info(
        f"Compacted {len(old_turns)} old turns into summary, "
        f"keeping {len(recent_turns)} recent turns"
    )
    return compacted


def check_context_budget(
    prompt_text: str,
    tokenizer,
    max_new_tokens: int,
) -> Tuple[bool, int, int]:
    """Check whether the prompt fits within the model's context window."""
    prompt_tokens = estimate_token_count(prompt_text, tokenizer)
    total_input = prompt_tokens + IMAGE_TOKEN_BUDGET
    available = MODEL_CONTEXT_LENGTH - total_input
    fits = available >= max_new_tokens
    return fits, total_input, max(0, available)


# ═══════════════════════════════════════════════════════════════
# Post-processing & quality checks
# ═══════════════════════════════════════════════════════════════

_ALL_DISEASE_NAMES = {
    "blast": ["rice blast", "blast"],
    "blight": ["bacterial blight", "blight"],
    "brownspot": ["brown spot", "brownspot", "brown_spot"],
}


def remove_cross_disease_contamination(text: str, primary_disease: Optional[str] = None) -> str:
    """
    Remove sentences that mention a DIFFERENT disease than the primary one.
    Fixes: 'This leaf has rice blast. The pattern fits moderate blight.'
    The second sentence contradicts the first and gets removed.
    """
    if not text:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text

    # Detect primary disease from first sentence if not provided
    first_sent_lower = sentences[0].lower()
    detected = primary_disease

    if not detected:
        for disease, keywords in _ALL_DISEASE_NAMES.items():
            if any(kw in first_sent_lower for kw in keywords):
                detected = disease
                break

    if not detected:
        return text

    # Collect keywords for OTHER diseases
    other_keywords = []
    for disease, keywords in _ALL_DISEASE_NAMES.items():
        if disease != detected:
            other_keywords.extend(keywords)

    # Filter out contaminated sentences
    clean_sentences = [sentences[0]]
    for sent in sentences[1:]:
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in other_keywords):
            logger.info(f"Removed cross-disease contamination: '{sent[:60]}...'")
            continue
        clean_sentences.append(sent)

    return " ".join(clean_sentences)


def clean_response(text: str, detected_disease: Optional[str] = None) -> str:
    """Clean and validate model output. Ensures no truncation or contamination."""
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Remove role leakage
    if "User:" in text:
        text = text.split("User:")[0].strip()
    if "Assistant:" in text:
        parts = text.split("Assistant:")
        text = parts[-1].strip() if len(parts) > 1 else text

    # Remove self-generated questions
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith("?") and any(
            stripped.lower().startswith(w)
            for w in ["what ", "how ", "when ", "where ", "why ", "should ", "can ", "do ", "is ", "are "]
        ):
            if not any(kw in stripped.lower() for kw in ["you should", "you can", "you need"]):
                logger.debug(f"Removed self-question: {stripped[:60]}")
                continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    # Remove bullet fragments
    text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)

    # Collapse newlines into spaces
    text = re.sub(r'\n+', ' ', text).strip()

    # Remove cross-disease contamination
    text = remove_cross_disease_contamination(text, detected_disease)

    # Anti-truncation: ensure response ends at sentence boundary
    if text and text[-1] not in ".!?":
        last_period = text.rfind(".")
        last_excl = text.rfind("!")
        last_quest = text.rfind("?")
        last_end = max(last_period, last_excl, last_quest)

        if last_end > 0:
            text = text[: last_end + 1]
        else:
            text = text.rstrip(",;:— -") + "."

    # Deduplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
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
# Token-level generation helpers
# ═══════════════════════════════════════════════════════════════

def apply_repetition_penalty_to_logits(
    logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.4
):
    """Apply repetition penalty to logits."""
    if penalty == 1.0:
        return logits
    last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
    for b in range(input_ids.shape[0]):
        unique_ids = input_ids[b].unique()
        score = last_logits[b, unique_ids]
        score = torch.where(score < 0, score * penalty, score / penalty)
        last_logits[b, unique_ids] = score
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling."""
    if temperature <= 0.0:
        return logits
    return logits / temperature


# ═══════════════════════════════════════════════════════════════
# Main generation function
# ═══════════════════════════════════════════════════════════════

def generate_answer(
    image: Image.Image,
    question: str,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
    max_new_tokens: Optional[int] = None,
    detected_disease: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Generate answer from the model given an image and question.
    
    Returns:
        Tuple of (response_text, context_status)
    """
    model, tokenizer, image_processor, vision_tower, mm_projector, embed_tokens, device = get_model()

    is_followup = conversation_history is not None and len(conversation_history) > 0
    context_status = "ok"

    # ─── Off-topic check BEFORE running inference (saves GPU time) ───
    if is_followup:
        if not is_follow_up_on_topic(question, detected_disease):
            logger.info(f"Off-topic follow-up blocked: '{question[:60]}'")
            return OFF_TOPIC_RESPONSE, "ok"
    else:
        if not is_question_on_topic(question):
            logger.info(f"Off-topic question blocked: '{question[:60]}'")
            return OFF_TOPIC_RESPONSE, "ok"

    # ─── Set generation budget ───
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS_FOLLOWUP if is_followup else MAX_NEW_TOKENS
    max_sentences = MAX_SENTENCES_FOLLOWUP if is_followup else MAX_SENTENCES_TURN1

    try:
        # ─── Process image ───
        pixel_values = image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device, dtype=torch.float32)

        # ─── Context compaction ───
        if is_followup and conversation_history:
            disease_label = detected_disease or "unknown"
            conversation_history = compact_conversation_history(
                conversation_history, disease_label, tokenizer
            )
            if len(conversation_history) > MAX_FULL_HISTORY_TURNS:
                context_status = "compacted"

        # ─── Build prompt ───
        prompt_parts = [f"System: {SYSTEM_PROMPT}\n"]

        # Anchor the disease for follow-ups to prevent cross-contamination
        if is_followup and detected_disease:
            prompt_parts.append(
                f"[The image shows rice {detected_disease}. Answer the follow-up question directly.]\n"
            )

        if conversation_history:
            for prev_q, prev_a in conversation_history:
                clean_q = prev_q.replace("<image>", "").strip()
                clean_a = prev_a.strip()
                if len(clean_a) > 150:
                    sents = re.split(r'(?<=[.!?])\s+', clean_a)
                    clean_a = " ".join(sents[:2])
                    if not clean_a.endswith((".", "!", "?")):
                        clean_a += "."
                prompt_parts.append(f"User: {clean_q}\nAssistant: {clean_a}\n")

        current_q = question.replace("<image>", "").strip()
        prompt_parts.append(f"User: {current_q}\nAssistant:")
        prompt = "".join(prompt_parts)

        # ─── Check context budget ───
        fits, input_tokens, available = check_context_budget(
            prompt, tokenizer, max_new_tokens
        )
        if not fits:
            if conversation_history and len(conversation_history) > 1:
                logger.warning(
                    f"Context overflow ({input_tokens}+{max_new_tokens} > {MODEL_CONTEXT_LENGTH}), "
                    f"trimming to last turn only"
                )
                conversation_history = conversation_history[-1:]
                context_status = "compacted"
                prompt_parts = [f"System: {SYSTEM_PROMPT}\n"]
                if detected_disease:
                    prompt_parts.append(
                        f"[The image shows rice {detected_disease}. Answer directly.]\n"
                    )
                for prev_q, prev_a in conversation_history:
                    prompt_parts.append(
                        f"User: {prev_q.replace('<image>', '').strip()}\n"
                        f"Assistant: {prev_a.strip()[:100]}\n"
                    )
                prompt_parts.append(f"User: {current_q}\nAssistant:")
                prompt = "".join(prompt_parts)

            _, input_tokens2, available2 = check_context_budget(
                prompt, tokenizer, max_new_tokens
            )
            if available2 < 20:
                context_status = "near_limit"
                max_new_tokens = max(20, available2)
                logger.warning(f"Near context limit, capping to {max_new_tokens} tokens")

        # ─── Get vision features ───
        with torch.no_grad(), torch.inference_mode():
            image_features = vision_tower(pixel_values.to(dtype=vision_tower.dtype))
            if hasattr(image_features, "last_hidden_state"):
                image_features = image_features.last_hidden_state
            elif isinstance(image_features, tuple):
                image_features = image_features[0]
            image_embeds = mm_projector(image_features)

        # ─── Tokenize and embed ───
        text_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        text_embeds = embed_tokens(text_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        prompt_len = inputs_embeds.shape[1]

        # ─── Generate tokens ───
        gen_token_ids = []
        past_kv = None

        with torch.no_grad(), torch.inference_mode():
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=torch.ones(1, prompt_len, device=device, dtype=torch.long),
                        use_cache=True,
                    )
                else:
                    outputs = model(
                        input_ids=next_token,
                        attention_mask=torch.ones(
                            1, prompt_len + len(gen_token_ids), device=device, dtype=torch.long
                        ),
                        past_key_values=past_kv,
                        use_cache=True,
                    )

                past_kv = outputs.past_key_values
                raw_logits = outputs.logits[:, -1, :].clone()

                # Repetition penalty
                if REPETITION_PENALTY > 1.0 and gen_token_ids:
                    seen = torch.tensor(gen_token_ids, device=device, dtype=torch.long).unsqueeze(0)
                    raw_logits = apply_repetition_penalty_to_logits(
                        raw_logits.unsqueeze(1), seen, REPETITION_PENALTY
                    ).squeeze(1)

                # N-gram blocking
                if GEN_NO_REPEAT_NGRAM > 1 and len(gen_token_ids) >= GEN_NO_REPEAT_NGRAM - 1:
                    ngram_prefix = tuple(gen_token_ids[-(GEN_NO_REPEAT_NGRAM - 1):])
                    banned = set()
                    for start in range(len(gen_token_ids) - (GEN_NO_REPEAT_NGRAM - 1)):
                        window = tuple(gen_token_ids[start: start + GEN_NO_REPEAT_NGRAM - 1])
                        if window == ngram_prefix:
                            banned.add(gen_token_ids[start + GEN_NO_REPEAT_NGRAM - 1])
                    if banned:
                        ban_ids = torch.tensor(list(banned), device=device, dtype=torch.long)
                        raw_logits[0, ban_ids] = float("-inf")

                # Temperature
                if TEMPERATURE > 0:
                    raw_logits = apply_temperature(raw_logits, TEMPERATURE)

                next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                gen_token_ids.append(next_token.item())

                # Early stopping at sentence boundary
                if len(gen_token_ids) >= MIN_TOKENS_FOR_EARLY_STOP:
                    current_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
                    sentence_endings = (
                        current_text.count(". ")
                        + current_text.count(".\n")
                        + current_text.count("! ")
                        + current_text.count("!\n")
                    )
                    if current_text.rstrip().endswith((".", "!", "?")):
                        sentence_endings += 1
                    if sentence_endings >= max_sentences:
                        break

        # ─── Decode and clean ───
        decoded = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
        decoded = clean_response(decoded, detected_disease)

        if not is_followup and decoded:
            logger.info(f"Turn-1 response: {decoded[:120]}")

        # Cleanup
        del inputs_embeds, image_embeds, text_embeds, pixel_values, past_kv, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return decoded, context_status

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise


# ═══════════════════════════════════════════════════════════════
# Disease extraction
# ═══════════════════════════════════════════════════════════════

def infer_disease(text: str) -> str:
    """Extract disease from model response. First sentence only."""
    if not text or not text.strip():
        return "unknown"

    t = text.lower().strip()
    first_sent = t.split(".")[0] if "." in t else t[:80]

    if "bacterial blight" in first_sent:
        return "blight"
    if "brown spot" in first_sent or "brownspot" in first_sent:
        return "brownspot"
    if "blast" in first_sent and "blight" not in first_sent:
        return "blast"
    if "blight" in first_sent:
        return "blight"

    chunk = t[:100]
    if "bacterial blight" in chunk:
        return "blight"
    if "brown spot" in chunk or "brownspot" in chunk:
        return "brownspot"
    if "blast" in chunk and "blight" not in chunk:
        return "blast"
    if "blight" in chunk:
        return "blight"

    return "unknown"