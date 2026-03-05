"""Pydantic schemas for API request/response validation."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    image: str = Field(..., description="Base64 encoded image string")
    question: str = Field(..., description="User's question about the rice disease")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Previous conversation messages for context"
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Maximum tokens to generate (auto-set based on turn)"
    )


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    response: str = Field(..., description="Model's response")
    disease_detected: Optional[str] = Field(
        default=None,
        description="Detected disease: 'blast', 'blight', 'brownspot', or None"
    )
    context_status: Optional[str] = Field(
        default=None,
        description="Context window status: 'ok', 'compacted', or 'near_limit'"
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used: 'cuda' or 'cpu'")
    model_path: str = Field(..., description="Path to loaded model checkpoint")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")