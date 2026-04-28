"""Pydantic schemas for API request/response validation."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class RatingCount(BaseModel):
    """Count of plants at a given 0-5 severity rating within one sample."""
    rating: int = Field(..., ge=0, le=5, description="Severity score 0-5")
    count: int = Field(..., ge=0, description="Number of plants with this rating")


class FieldSample(BaseModel):
    """One quadrat / plot sample: total vs infected plants, optional rating breakdown."""
    quadrant_id: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        description="Field quadrant 1-4 (optional)",
    )
    total_plants: int = Field(..., gt=0, description="Total plants in this sample")
    infected_plants: int = Field(..., ge=0, description="Infected plants in this sample")
    rating_counts: Optional[List[RatingCount]] = Field(
        default=None,
        description="Optional breakdown for severity index (counts must sum to total_plants)",
    )

    @model_validator(mode="after")
    def infected_lte_total(self) -> "FieldSample":
        if self.infected_plants > self.total_plants:
            raise ValueError("infected_plants cannot exceed total_plants")
        if self.rating_counts is not None:
            s = sum(rc.count for rc in self.rating_counts)
            if s != self.total_plants:
                raise ValueError(
                    "Sum of rating_counts must equal total_plants for each sample"
                )
        return self


# Area units for field size; incidence math keeps values in the user-chosen unit.
AreaUnit = Literal["m2", "kanal", "marla", "acre", "ha"]


class FieldSamplingReport(BaseModel):
    """Structured field sampling data for incidence and area estimates."""
    total_field_area: float = Field(..., gt=0, description="Total field area (see area_unit)")
    area_unit: AreaUnit = Field(..., description="Unit of total_field_area")
    quadrat_radius_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Radius in meters if using a circular quadrat (for area pi r^2)",
    )
    samples: List[FieldSample] = Field(
        ...,
        min_length=1,
        description="At least one sample (quadrat) with plant counts",
    )
    detected_disease: Optional[str] = Field(
        default=None,
        description="Disease label (blast/blight/brownspot) for disease-aware tier classification",
    )


class FieldMetricsState(BaseModel):
    """Client-roundtripped state: sampling guidance + last field-level estimates."""

    sampling_guidance_offered: bool = Field(
        default=False,
        description="True after the app appended quadrat sampling instructions once",
    )
    last_avg_incidence_pct: Optional[float] = Field(
        default=None,
        description="Most recent average disease incidence across samples (%)",
    )
    last_num_samples: Optional[int] = Field(
        default=None,
        description="Number of samples in last calculation",
    )
    last_estimated_infected_area_display: Optional[str] = Field(
        default=None,
        description="Human-readable estimated infected area, e.g. 2.1 ha",
    )
    last_avg_severity_pct: Optional[float] = Field(
        default=None,
        description="Field severity percentage used for whole-field severity banding",
    )
    last_estimated_infected_area_ha: Optional[float] = Field(
        default=None,
        description="Most recent estimated infected area converted to hectares",
    )
    last_field_area_ha: Optional[float] = Field(
        default=None,
        description="Most recent total field area converted to hectares",
    )
    field_severity_tier: Optional[Literal["low", "moderate", "high", "critical"]] = Field(
        default=None,
        description="Field severity tier matching VQA training: low/moderate/high/critical",
    )
    last_detected_disease: Optional[Literal["blast", "blight", "brownspot"]] = Field(
        default=None,
        description="Most recently detected disease label for field context injection",
    )


class FieldMetricsResult(BaseModel):
    """Structured output from field sampling calculations."""
    per_sample_incidence_pct: List[float] = Field(
        ...,
        description="Disease incidence per sample (%)",
    )
    average_incidence_pct: float = Field(..., description="Mean incidence across samples")
    num_samples: int = Field(..., ge=1)
    estimated_infected_area: float = Field(
        ...,
        ge=0,
        description="Estimated area with disease presence (same unit as input field area)",
    )
    estimated_infected_area_unit: AreaUnit = Field(
        ...,
        description="Unit of estimated_infected_area and total_field_area",
    )
    total_field_area: float = Field(..., gt=0)
    quadrat_area_m2: Optional[float] = Field(
        default=None,
        description="Quadrat area in m^2 if quadrat_radius_m was given",
    )
    per_sample_severity_pct: Optional[List[float]] = Field(
        default=None,
        description="Per-sample severity index (%) when rating_counts provided",
    )
    average_severity_pct: Optional[float] = Field(
        default=None,
        description="Mean severity index (%) across samples with ratings",
    )
    narrative: str = Field(..., description="Plain-language summary for the user")


class FieldMetricsCalculateResponse(BaseModel):
    """Response from standalone field metrics calculator (no VLM)."""
    field_metrics: FieldMetricsResult = Field(..., description="Computed metrics and narrative")
    field_metrics_state: FieldMetricsState = Field(
        ..., description="State to merge into chat context on return"
    )


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    image: str = Field(..., description="Base64 encoded image string")
    question: str = Field(..., description="User's question about the rice disease")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Previous conversation messages for context",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=128,
        description="Maximum tokens to generate (server also applies HARD_CAP_NEW_TOKENS)",
    )
    field_metrics_state: Optional[FieldMetricsState] = Field(
        default=None,
        description="Echo prior state from last response for compaction and guidance flags",
    )


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    response: str = Field(..., description="Model's response")
    disease_detected: Optional[str] = Field(
        default=None,
        description="Detected disease: 'blast', 'blight', 'brownspot', or None",
    )
    context_status: Optional[str] = Field(
        default=None,
        description="Context window status: 'ok', 'compacted', or 'near_limit'",
    )
    image_rejected: bool = Field(
        default=False,
        description="True when the uploaded image was rejected as non-plant (Guard 2)",
    )
    image_quality_issue: Optional[str] = Field(
        default=None,
        description="Optional reason when image is rejected for quality (e.g. blurry/too dark)",
    )
    field_metrics: Optional[FieldMetricsResult] = Field(
        default=None,
        description="Deprecated for chat; field metrics come from /api/field-metrics/calculate only",
    )
    field_metrics_state: Optional[FieldMetricsState] = Field(
        default=None,
        description="Updated state to send back on the next request",
    )
    sampling_guidance_appended: bool = Field(
        default=False,
        description="True when the client should show the field survey CTA (model text unchanged)",
    )
    field_survey_cta: bool = Field(
        default=False,
        description="When True, show one-line field survey prompt and button in the chat UI",
    )
    field_survey_cta_text: Optional[str] = Field(
        default=None,
        description="Short line to render with the field survey button when field_survey_cta is True",
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