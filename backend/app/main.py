"""FastAPI application for Rice Disease Detection Chatbot."""
import logging
import time
import asyncio
from collections import defaultdict
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.config import (
    MODEL_PATH,
    HOST,
    PORT,
    OFF_TOPIC_RESPONSE,
    OFF_TOPIC_QUESTION_RESPONSE,
    FIELD_SURVEY_CTA_TEXT,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ErrorResponse,
    ConversationMessage,
    FieldMetricsState,
    FieldSamplingReport,
    FieldMetricsCalculateResponse,
)
from app.model_loader import load_model, get_device
from app.field_metrics import (
    classify_field_severity_band,
    compute_field_metrics,
    convert_area_to_ha,
    metrics_result_to_state_display,
)
from app.inference import (
    generate_answer,
    decode_base64_image,
    infer_disease,
    maybe_append_sampling_guidance,
    detect_image_quality_issue,
    get_image_quality_rejection_response,
    get_image_quality_issue_label,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rice Disease Detection API",
    description="FastAPI backend for FastVLM 1.5B rice disease detection chatbot",
    version="2.0.0",
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
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


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
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.post(
    "/api/field-metrics/calculate",
    response_model=FieldMetricsCalculateResponse,
    tags=["Field metrics"],
)
async def calculate_field_metrics_endpoint(report: FieldSamplingReport):
    """
    Compute disease incidence and estimated affected area from quadrat samples.
    Does not require the vision model; safe to call when the model is still loading.
    """
    try:
        result = compute_field_metrics(report)
        prev = FieldMetricsState()
        infected_area_ha = convert_area_to_ha(
            result.estimated_infected_area, result.estimated_infected_area_unit
        )
        total_area_ha = convert_area_to_ha(result.total_field_area, result.estimated_infected_area_unit)

        severity_idx = round(
            result.average_severity_pct
            if result.average_severity_pct is not None
            else result.average_incidence_pct,
            1,
        )

        disease_label = getattr(report, "detected_disease", None) or "blast"

        new_state = FieldMetricsState(
            sampling_guidance_offered=prev.sampling_guidance_offered,
            last_avg_incidence_pct=result.average_incidence_pct,
            last_num_samples=result.num_samples,
            last_estimated_infected_area_display=metrics_result_to_state_display(result),
            last_avg_severity_pct=severity_idx,
            last_estimated_infected_area_ha=round(infected_area_ha, 4),
            last_field_area_ha=round(total_area_ha, 4),
            field_severity_tier=classify_field_severity_band(
                incidence_pct=result.average_incidence_pct,
                severity_idx=severity_idx,
                disease=disease_label,
            ),
        )
        return FieldMetricsCalculateResponse(
            field_metrics=result,
            field_metrics_state=new_state,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    device = get_device()
    return HealthResponse(
        status="healthy" if _model_loaded else "unhealthy",
        model_loaded=_model_loaded,
        device=str(device),
        model_path=MODEL_PATH,
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint for rice disease detection.
    
    Accepts a base64 image and question, returns model response.
    Handles conversation history with automatic context compaction.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    session_id = (http_request.headers.get("x-chat-session-id") or "anonymous").strip() or "anonymous"
    session_lock = _session_locks[session_id]

    async with session_lock:
        try:
            start_time = time.time()

            # Parse conversation history into tuples (shared by VLM and field-metrics paths)
            conversation_history = None
            detected_disease = None

            if request.conversation_history:
                history_tuples = []
                history = request.conversation_history
                i = 0
                while i < len(history):
                    if history[i].role == "user":
                        if i + 1 < len(history) and history[i + 1].role == "assistant":
                            history_tuples.append(
                                (history[i].content, history[i + 1].content)
                            )
                            if detected_disease is None:
                                detected_disease = infer_disease(history[i + 1].content)
                            i += 2
                        else:
                            logger.warning(f"Skipping unpaired user message at index {i}")
                            i += 1
                    else:
                        logger.warning(f"Skipping orphan assistant message at index {i}")
                        i += 1

                conversation_history = history_tuples if history_tuples else None
                logger.info(
                    f"Parsed {len(history_tuples)} history pairs from {len(history)} messages, "
                    f"detected disease from history: {detected_disease}"
                )

            image = decode_base64_image(request.image)
            fms = request.field_metrics_state
            quality_issue = detect_image_quality_issue(image)
            if quality_issue is not None:
                return ChatResponse(
                    response=get_image_quality_rejection_response(),
                    disease_detected=None,
                    context_status="ok",
                    image_rejected=True,
                    image_quality_issue=get_image_quality_issue_label(quality_issue),
                    field_metrics=None,
                    field_metrics_state=fms,
                    sampling_guidance_appended=False,
                    field_survey_cta=False,
                    field_survey_cta_text=None,
                )

            # Ensure the field_metrics_state has the disease label before inference
            # so _build_field_context_prefix can use it even if detected_disease is None
            if fms and detected_disease and detected_disease in ("blast", "blight", "brownspot"):
                if fms.last_detected_disease is None:
                    fms = fms.model_copy(update={"last_detected_disease": detected_disease})

            response, context_status = generate_answer(
                image=image,
                question=request.question,
                conversation_history=conversation_history,
                max_new_tokens=request.max_new_tokens,
                detected_disease=detected_disease,
                field_metrics_state=fms,
            )

            response, guidance_appended, metrics_state_out, field_survey_cta = (
                maybe_append_sampling_guidance(
                    request.question,
                    response,
                    fms,
                )
            )

            disease_detected = infer_disease(response)
            if metrics_state_out and disease_detected in ("blast", "blight", "brownspot"):
                metrics_state_out = metrics_state_out.model_copy(
                    update={"last_detected_disease": disease_detected}
                )

            image_rejected = response == OFF_TOPIC_RESPONSE

            if response in (OFF_TOPIC_RESPONSE, OFF_TOPIC_QUESTION_RESPONSE):
                disease_detected = "unknown"

            inference_time = time.time() - start_time
            logger.info(
                f"Inference completed in {inference_time:.2f}s | "
                f"disease={disease_detected} | context={context_status} | "
                f"image_rejected={image_rejected} | field_survey_cta={field_survey_cta} | "
                f"response_len={len(response)}"
            )

            return ChatResponse(
                response=response,
                disease_detected=disease_detected if disease_detected != "unknown" else None,
                context_status=context_status,
                image_rejected=image_rejected,
                image_quality_issue=None,
                field_metrics=None,
                field_metrics_state=metrics_state_out,
                sampling_guidance_appended=guidance_appended,
                field_survey_cta=field_survey_cta,
                field_survey_cta_text=FIELD_SURVEY_CTA_TEXT if field_survey_cta else None,
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/upload", response_model=ChatResponse, tags=["Chat"])
async def chat_with_upload(
    file: UploadFile = File(...),
    question: str = Form(default="What disease does this rice leaf have?"),
    max_new_tokens: Optional[int] = Form(default=None),
):
    """
    Alternative endpoint that accepts image file upload instead of base64.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        quality_issue = detect_image_quality_issue(image)
        if quality_issue is not None:
            return ChatResponse(
                response=get_image_quality_rejection_response(),
                disease_detected=None,
                context_status="ok",
                image_rejected=True,
                image_quality_issue=get_image_quality_issue_label(quality_issue),
            )

        start_time = time.time()
        response, context_status = generate_answer(
            image=image,
            question=question,
            conversation_history=None,
            max_new_tokens=max_new_tokens,
        )

        disease_detected = infer_disease(response)

        image_rejected = response == OFF_TOPIC_RESPONSE

        if response in (OFF_TOPIC_RESPONSE, OFF_TOPIC_QUESTION_RESPONSE):
            disease_detected = "unknown"

        inference_time = time.time() - start_time
        logger.info(f"Upload inference completed in {inference_time:.2f}s")

        return ChatResponse(
            response=response,
            disease_detected=disease_detected if disease_detected != "unknown" else None,
            context_status=context_status,
            image_rejected=image_rejected,
            image_quality_issue=None,
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
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)