"""Inference pipeline for generating responses from the model."""
import base64
import io
import logging
import torch
from typing import List, Optional, Tuple
from PIL import Image

from app.config import MAX_NEW_TOKENS, REPETITION_PENALTY, GEN_NO_REPEAT_NGRAM, MIN_TOKENS_FOR_EARLY_STOP
from app.model_loader import get_model

logger = logging.getLogger(__name__)

IMAGE_TOKEN_INDEX = -200


def decode_base64_image(image_string: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if "," in image_string:
            image_string = image_string.split(",")[1]
        
        image_data = base64.b64decode(image_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {str(e)}")


def apply_repetition_penalty_to_logits(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.3):
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


def generate_answer(
    image: Image.Image,
    question: str,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """
    Generate answer from the model given an image and question.
    
    Args:
        image: PIL Image object
        question: User's question
        conversation_history: List of (question, answer) tuples for context
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated response string
    """
    model, tokenizer, image_processor, vision_tower, mm_projector, embed_tokens, device = get_model()
    
    # Determine if this is a follow-up turn (has conversation history)
    is_followup = conversation_history is not None and len(conversation_history) > 0
    # Allow more sentences for follow-ups since they need to answer contextually
    max_sentences = 4 if is_followup else 3
    # Use higher token limit for follow-ups
    if is_followup and max_new_tokens < 150:
        max_new_tokens = 150
    
    try:
        # Process image
        pixel_values = image_processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].to(device, dtype=torch.float32)
        
        # Build prompt from conversation history
        prompt_parts = []
        if conversation_history:
            for prev_q, prev_a in conversation_history:
                clean_q = prev_q.replace("<image>", "").strip()
                prompt_parts.append(f"User: {clean_q}\nAssistant: {prev_a}\n")
        
        current_q = question.replace("<image>", "").strip()
        prompt_parts.append(f"User: {current_q}\nAssistant: ")
        prompt = "".join(prompt_parts)
        
        # Get vision features
        with torch.no_grad(), torch.inference_mode():
            image_features = vision_tower(pixel_values.to(dtype=vision_tower.dtype))
            
            # Extract hidden state
            if hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state
            elif isinstance(image_features, tuple):
                image_features = image_features[0]
            
            # Project through mm_projector
            image_embeds = mm_projector(image_features)
        
        # Tokenize and embed text
        text_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        text_embeds = embed_tokens(text_ids)
        
        # Concatenate image and text embeddings
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        prompt_len = inputs_embeds.shape[1]
        
        # Generate tokens
        gen_token_ids = []
        past_kv = None
        
        with torch.no_grad(), torch.inference_mode():
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=torch.ones(1, prompt_len, device=device, dtype=torch.long),
                        use_cache=True
                    )
                else:
                    outputs = model(
                        input_ids=next_token,
                        attention_mask=torch.ones(1, prompt_len + len(gen_token_ids), device=device, dtype=torch.long),
                        past_key_values=past_kv,
                        use_cache=True
                    )
                
                past_kv = outputs.past_key_values
                raw_logits = outputs.logits[:, -1, :]
                
                # Apply repetition penalty
                if REPETITION_PENALTY > 1.0 and gen_token_ids:
                    seen = torch.tensor(gen_token_ids, device=device, dtype=torch.long).unsqueeze(0)
                    raw_logits = apply_repetition_penalty_to_logits(
                        raw_logits.unsqueeze(1),
                        seen,
                        REPETITION_PENALTY
                    ).squeeze(1)
                
                # Apply n-gram blocking
                if GEN_NO_REPEAT_NGRAM > 1 and len(gen_token_ids) >= GEN_NO_REPEAT_NGRAM - 1:
                    ngram_prefix = tuple(gen_token_ids[-(GEN_NO_REPEAT_NGRAM - 1):])
                    banned = set()
                    for start in range(len(gen_token_ids) - (GEN_NO_REPEAT_NGRAM - 1)):
                        if tuple(gen_token_ids[start:start + GEN_NO_REPEAT_NGRAM - 1]) == ngram_prefix:
                            banned.add(gen_token_ids[start + GEN_NO_REPEAT_NGRAM - 1])
                    if banned:
                        ban_ids = torch.tensor(list(banned), device=device, dtype=torch.long)
                        raw_logits[0, ban_ids] = float("-inf")
                
                # Sample next token (greedy)
                next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)
                
                # Stop on EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                gen_token_ids.append(next_token.item())
                
                # Early stopping: stop after complete sentences for concise answers
                if len(gen_token_ids) >= MIN_TOKENS_FOR_EARLY_STOP:
                    # Decode current tokens to check for sentence completion
                    current_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
                    # Count complete sentences (period followed by space or newline)
                    sentence_endings = current_text.count('. ') + current_text.count('.\n')
                    # Allow more sentences for follow-up turns
                    if sentence_endings >= max_sentences:
                        break
        
        # Decode response
        decoded = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
        
        # Clean response (remove "User:" if present)
        if "User:" in decoded:
            decoded = decoded.split("User:")[0].strip()
        
        # Truncate to reasonable length for concise answers
        sentences = decoded.split('. ')
        max_post_sentences = max_sentences + 1  # Allow slightly more than early stop
        if len(sentences) > max_post_sentences:
            decoded = '. '.join(sentences[:max_post_sentences])
            if not decoded.endswith('.'):
                decoded += '.'
        
        # Cleanup
        del inputs_embeds, image_embeds, text_embeds, pixel_values, past_kv, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return decoded
    
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise


def infer_disease(text: str) -> str:
    """
    Extract disease classification from model response.
    Uses first sentence only to avoid bias from later mentions.
    
    Args:
        text: Model response text
    
    Returns:
        Disease label: 'blast', 'blight', 'brownspot', or 'unknown'
    """
    if not text or not text.strip():
        return "unknown"
    
    t = text.lower().strip()
    
    # Extract first sentence
    first_sent = t.split('.')[0] if '.' in t else t[:100]
    
    # Check most specific patterns first
    if 'bacterial blight' in first_sent:
        return 'blight'
    if 'brown spot' in first_sent or 'brownspot' in first_sent or 'brown_spot' in first_sent:
        return 'brownspot'
    if 'blast' in first_sent and 'blight' not in first_sent:
        return 'blast'
    if 'blight' in first_sent:
        return 'blight'
    
    # Fallback: check first 150 chars
    first_chunk = t[:150]
    if 'bacterial blight' in first_chunk:
        return 'blight'
    if 'brown spot' in first_chunk or 'brownspot' in first_chunk:
        return 'brownspot'
    if 'blast' in first_chunk and 'blight' not in first_chunk:
        return 'blast'
    if 'blight' in first_chunk:
        return 'blight'
    
    return "unknown"