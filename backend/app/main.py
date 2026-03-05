"""FastAPI application for Rice Disease Detection Chatbot."""
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.config import MODEL_PATH, HOST, PORT
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ErrorResponse,
    ConversationMessage
)
from app.model_loader import load_model, get_device
from app.inference import generate_answer, decode_base64_image, infer_disease

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rice Disease Detection API",
    description="FastAPI backend for FastVLM 1.5B rice disease detection chatbot",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loading flag
_model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global _model_loaded
    logger.info("Starting up application...")
    try:
        start_time = time.time()
        load_model()
        load_time = time.time() - start_time
        _model_loaded = True
        device = get_device()
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Device: {device}")
        logger.info(f"Model path: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        _model_loaded = False
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Rice Disease Detection API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    device = get_device()
    return HealthResponse(
        status="healthy" if _model_loaded else "unhealthy",
        model_loaded=_model_loaded,
        device=str(device),
        model_path=MODEL_PATH
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint for rice disease detection.
    
    Accepts an image (base64 encoded) and a question, returns model response.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Decode image
        start_time = time.time()
        image = decode_base64_image(request.image)
        
        # Convert conversation history to list of tuples
        conversation_history = None
        if request.conversation_history:
            # Filter to only valid user/assistant pairs
            history_tuples = []
            history = request.conversation_history
            i = 0
            while i < len(history):
                # Find next user message
                if history[i].role == "user":
                    # Look for the matching assistant response
                    if i + 1 < len(history) and history[i + 1].role == "assistant":
                        history_tuples.append((
                            history[i].content,
                            history[i + 1].content
                        ))
                        i += 2
                    else:
                        # Unpaired user message, skip it
                        logger.warning(f"Skipping unpaired user message at index {i}")
                        i += 1
                else:
                    # Skip orphan assistant messages
                    logger.warning(f"Skipping orphan assistant message at index {i}")
                    i += 1
            
            conversation_history = history_tuples if history_tuples else None
            logger.info(f"Parsed {len(history_tuples)} history pairs from {len(history)} messages")
        
        # Generate response (use smaller default for concise answers)
        response = generate_answer(
            image=image,
            question=request.question,
            conversation_history=conversation_history,
            max_new_tokens=request.max_new_tokens or 100
        )
        
        # Extract disease if present
        disease_detected = infer_disease(response)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        return ChatResponse(
            response=response,
            disease_detected=disease_detected if disease_detected != "unknown" else None
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/upload", response_model=ChatResponse, tags=["Chat"])
async def chat_with_upload(
    file: UploadFile = File(...),
    question: str = "What disease does this rice leaf have?",
    max_new_tokens: int = 200
):
    """
    Alternative endpoint that accepts image file upload instead of base64.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Read and decode image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Generate response (cap at 100 for concise answers)
        start_time = time.time()
        response = generate_answer(
            image=image,
            question=question,
            conversation_history=None,
            max_new_tokens=min(max_new_tokens, 100)
        )
        
        # Extract disease
        disease_detected = infer_disease(response)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        return ChatResponse(
            response=response,
            disease_detected=disease_detected if disease_detected != "unknown" else None
        )
    
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)