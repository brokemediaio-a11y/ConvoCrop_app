# Quick Start Guide

## Step 1: Create and Activate Virtual Environment

**Windows:**
```bash
# Create venv
python -m venv venv

# Activate venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with PyTorch installation, install it separately first:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install -r requirements.txt
```

## Step 3: Verify Model Checkpoint

Make sure your model checkpoint is in:
```
backend/model/checkpoints/best/
```

Required files:
- `adapter_model.safetensors`
- `adapter_config.json`
- `mm_projector.pt`
- Tokenizer files (vocab.json, tokenizer_config.json, etc.)

## Step 4: Run the Server

```bash
python run_server.py
```

The server will:
1. Load the model (takes ~30-60 seconds on GPU)
2. Start the API server on http://localhost:8000
3. Show API docs at http://localhost:8000/docs

## Step 5: Test the API

### Using the API Docs (Recommended)
1. Open http://localhost:8000/docs in your browser
2. Try the `/api/health` endpoint first to verify model is loaded
3. Use `/api/chat` endpoint with a base64-encoded image

### Using curl

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Chat (with base64 image):**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "question": "What disease does this rice leaf have?"
  }'
```

## Troubleshooting

### Model Loading Takes Too Long
- First load is always slow (~30-60s on GPU)
- Check GPU is being used: `nvidia-smi` (Windows) or check `/api/health`

### CUDA Out of Memory
- The model uses ~4-5GB VRAM on GPU
- Ensure no other processes are using GPU
- Try restarting the server

### Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Model Not Found
- Check `backend/model/checkpoints/best/` exists
- Verify `mm_projector.pt` is present
- Check logs for specific error messages

## Next Steps

Once the backend is running, you can:
1. Test it with the API docs interface
2. Build the frontend to connect to this API
3. The frontend will call `/api/chat` endpoint with images

## Performance Expectations

**With RTX 4050 Ti (6GB VRAM):**
- Model loading: 30-60 seconds
- First inference: 5-10 seconds (warmup)
- Subsequent inferences: 2-5 seconds per request

**On CPU:**
- Model loading: 2-5 minutes
- Inference: 30 seconds - 2 minutes per request
