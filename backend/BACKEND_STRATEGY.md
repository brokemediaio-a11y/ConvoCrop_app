# Backend Strategy: FastVLM 1.5B Rice Disease Detection API

## Overview
FastAPI backend for serving the finetuned FastVLM 1.5B model for rice disease detection. The model uses LoRA adapters and requires special handling for the mm_projector component.

## Architecture

### 1. Model Components
- **Base Model**: `apple/FastVLM-1.5B` (from HuggingFace)
- **LoRA Adapter**: `backend/model/checkpoints/best/adapter_model.safetensors`
- **mm_projector weights**: `backend/model/checkpoints/best/mm_projector.pt`
- **Tokenizer**: From checkpoint directory
- **Vision Tower**: Frozen, part of base model
- **Image Processor**: CLIPImageProcessor (512x512)

### 2. Model Loading Strategy

#### Initialization Flow:
1. Load base model with `torch.float32` (or `float16` if GPU available)
2. Load LoRA adapter using `PeftModel.from_pretrained()`
3. Load mm_projector weights from `mm_projector.pt` and inject into model
4. Extract components: `vision_tower`, `mm_projector`, `embed_tokens`
5. Set model to eval mode
6. Keep model in memory (singleton pattern)

#### Memory Optimization:
- **GPU Available**: Use `float16` precision, CUDA device
- **CPU Only**: Use `float32`, consider CPU optimizations
- **Quantization**: Optional 8-bit quantization if memory constrained (using bitsandbytes)
- **Lazy Loading**: Load model only once at startup, reuse for all requests

### 3. Inference Pipeline

Based on `generate_answer()` function from finetuning script:

1. **Image Processing**:
   - Receive image (base64 or file upload)
   - Convert to PIL Image (RGB)
   - Process with `image_processor` → pixel_values tensor

2. **Vision Encoding**:
   - Pass pixel_values through `vision_tower`
   - Extract `last_hidden_state` or first element if tuple
   - Project through `mm_projector` → image_embeds

3. **Text Processing**:
   - Build prompt from conversation history + current question
   - Format: `"User: {question}\nAssistant: "` (with history if present)
   - Tokenize with tokenizer → text_ids
   - Embed with `embed_tokens` → text_embeds

4. **Generation**:
   - Concatenate: `[image_embeds, text_embeds]` → inputs_embeds
   - Generate tokens autoregressively with repetition penalty
   - Stop on EOS token or max_new_tokens (default: 200)
   - Decode response

5. **Post-processing**:
   - Clean response (remove "User:" if present)
   - Clear CUDA cache
   - Return response

### 4. API Endpoints

#### POST `/api/chat`
Main chat endpoint for rice disease detection.

**Request**:
```json
{
  "image": "base64_encoded_image_string",
  "question": "What disease does this rice leaf have?",
  "conversation_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ],
  "max_new_tokens": 200
}
```

**Response**:
```json
{
  "response": "The rice leaf shows symptoms of bacterial blight...",
  "disease_detected": "blight",
  "confidence": "high"
}
```

#### GET `/api/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda" or "cpu"
}
```

#### POST `/api/upload` (Optional)
Alternative endpoint for file upload instead of base64.

### 5. File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── model_loader.py      # Model loading and initialization
│   ├── inference.py         # Inference pipeline
│   ├── schemas.py           # Pydantic models for request/response
│   └── config.py            # Configuration settings
├── model/
│   └── checkpoints/
│       └── best/            # Model checkpoint files
├── requirements.txt
├── .env                     # Environment variables (optional)
├── venv/                    # Virtual environment (gitignored)
└── run_server.py           # Server startup script
```

### 6. Dependencies

Core dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch
- `transformers` - HuggingFace transformers
- `peft` - LoRA adapter loading
- `pillow` - Image processing
- `numpy` - Numerical operations
- `pydantic` - Request/response validation
- `python-multipart` - File upload support

Optional optimizations:
- `bitsandbytes` - 8-bit quantization (if needed)
- `accelerate` - Model loading optimizations

### 7. Configuration

Environment variables or config file:
- `MODEL_PATH`: Path to checkpoint directory (default: `./model/checkpoints/best`)
- `BASE_MODEL_NAME`: Base model name (default: `apple/FastVLM-1.5B`)
- `DEVICE`: `cuda` or `cpu` (auto-detect if not set)
- `MAX_NEW_TOKENS`: Default max generation tokens (default: 200)
- `REPETITION_PENALTY`: Generation repetition penalty (default: 1.3)
- `USE_HALF_PRECISION`: Use float16 if GPU available (default: True)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)

### 8. Error Handling

- Model loading errors → Return 500 with clear message
- Invalid image format → Return 400 with validation error
- Generation failures → Return 500, log error, clear cache
- CUDA OOM → Fallback to CPU or return error with suggestion

### 9. Performance Optimizations for Local Demo

1. **Model Loading**:
   - Load once at startup (singleton)
   - Use `device_map="auto"` if multiple GPUs
   - Enable `torch.compile()` if PyTorch 2.0+ (optional)

2. **Inference**:
   - Use `torch.no_grad()` for all inference
   - Clear CUDA cache after each request
   - Use `torch.inference_mode()` context
   - Batch size = 1 (single requests)

3. **Memory Management**:
   - Explicit garbage collection after inference
   - Clear intermediate tensors
   - Use `torch.cuda.empty_cache()` frequently

4. **CPU Fallback**:
   - If GPU unavailable, use CPU with optimizations
   - Consider model quantization for CPU
   - Use `torch.set_num_threads()` for CPU parallelism

### 10. Virtual Environment Setup

1. Create venv: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
3. Install dependencies: `pip install -r requirements.txt`
4. Verify: `python -c "import torch; print(torch.__version__)"`

### 11. Startup Sequence

1. Check CUDA availability
2. Load configuration
3. Initialize model loader
4. Load model (with progress logging)
5. Verify model components
6. Start FastAPI server
7. Log ready status

### 12. Testing Strategy

- Unit tests for inference pipeline
- Integration tests for API endpoints
- Load testing for concurrent requests
- Memory profiling

### 13. Logging

- Model loading progress
- Request/response logging (without sensitive data)
- Error logging with stack traces
- Performance metrics (inference time, memory usage)

### 14. Security Considerations

- Input validation (image size limits, file type checks)
- Rate limiting (optional, for demo may not be needed)
- CORS configuration for frontend
- Request timeout handling

## Implementation Priority

1. **Phase 1**: Basic FastAPI setup + model loading
2. **Phase 2**: Inference pipeline implementation
3. **Phase 3**: API endpoints + error handling
4. **Phase 4**: Optimization + testing
5. **Phase 5**: Documentation + deployment scripts

## Notes for Demo

- Model is heavy (~1.5B parameters), expect 2-5GB VRAM usage
- First request may be slower (warmup)
- Consider showing loading state in frontend
- If GPU unavailable, CPU inference will be slow (30s-2min per request)
- Recommend at least 8GB RAM for CPU inference
