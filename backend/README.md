# Rice Disease Detection API - Backend

FastAPI backend for serving the finetuned FastVLM 1.5B model for rice disease detection.

## Setup Instructions

### 1. Create Virtual Environment

**Windows:**
```bash
# Option 1: Use the setup script
setup_venv.bat

# Option 2: Manual setup
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install PyTorch. If you need a specific CUDA version, install PyTorch separately first:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install -r requirements.txt
```

### 3. Verify Model Checkpoint

Ensure your model checkpoint is in the correct location:
```
backend/
└── model/
    └── checkpoints/
        └── best/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── mm_projector.pt
            └── ... (other tokenizer files)
```

### 4. Run the Server

```bash
python run_server.py
```

Or directly with uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## API Endpoints

### POST `/api/chat`

Main chat endpoint for rice disease detection.

**Request:**
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

**Response:**
```json
{
  "response": "The rice leaf shows symptoms of bacterial blight...",
  "disease_detected": "blight"
}
```

### POST `/api/upload`

Alternative endpoint that accepts image file upload.

**Form Data:**
- `file`: Image file (JPEG, PNG, etc.)
- `question`: User's question (optional, default: "What disease does this rice leaf have?")
- `max_new_tokens`: Maximum tokens to generate (optional, default: 200)

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "./model/checkpoints/best"
}
```

## Configuration

You can configure the server using environment variables:

- `BASE_MODEL_NAME`: Base model name (default: `apple/FastVLM-1.5B`)
- `MODEL_PATH`: Path to checkpoint directory (default: `./model/checkpoints/best`)
- `DEVICE`: Device to use - `auto`, `cuda`, or `cpu` (default: `auto`)
- `USE_HALF_PRECISION`: Use float16 on GPU (default: `true`)
- `MAX_NEW_TOKENS`: Default max generation tokens (default: `200`)
- `REPETITION_PENALTY`: Repetition penalty (default: `1.3`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)

Example:
```bash
# Windows
set DEVICE=cuda
set USE_HALF_PRECISION=true
python run_server.py

# Linux/Mac
export DEVICE=cuda
export USE_HALF_PRECISION=true
python run_server.py
```

## Performance Notes

### GPU (RTX 4050 Ti - 6GB VRAM)
- Model loading: ~30-60 seconds
- First inference: ~5-10 seconds (warmup)
- Subsequent inferences: ~2-5 seconds
- Memory usage: ~4-5GB VRAM

### CPU
- Model loading: ~2-5 minutes
- Inference: ~30 seconds - 2 minutes per request
- Memory usage: ~6-8GB RAM

## Troubleshooting

### CUDA Out of Memory
If you get CUDA OOM errors:
1. Ensure `USE_HALF_PRECISION=true` (uses float16)
2. Reduce `MAX_NEW_TOKENS` in requests
3. Restart the server to clear cache

### Model Not Loading
- Check that model checkpoint files exist in `model/checkpoints/best/`
- Verify `mm_projector.pt` is present
- Check logs for specific error messages

### Slow Inference
- Ensure GPU is being used (check `/api/health` endpoint)
- First request is always slower (warmup)
- CPU inference is significantly slower than GPU

## Development

### Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model_loader.py      # Model loading
│   ├── inference.py         # Inference pipeline
│   ├── schemas.py           # Pydantic models
│   └── config.py            # Configuration
├── model/
│   └── checkpoints/
│       └── best/            # Model checkpoint
├── requirements.txt
├── run_server.py
└── README.md
```

## License

See main project license.
