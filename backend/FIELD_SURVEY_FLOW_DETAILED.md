# Field Survey Flow (Backend Deep Dive)

This document explains how the **Field Survey** option is implemented in the backend, including:

- what triggers the field survey text/button in chat
- how the API service works
- what payloads are accepted
- what formulas are used
- what responses are returned

---

## 1) High-Level Backend Architecture

The backend is a FastAPI app (`backend/app/main.py`) with two relevant paths:

1. `POST /api/chat`
   - Runs image + question through the VLM pipeline.
   - Decides whether the frontend should show a **Field Survey CTA** (text + button).
2. `POST /api/field-metrics/calculate`
   - Computes deterministic field-level disease metrics from quadrat samples.
   - Does not need model inference.

Key modules:

- `backend/app/main.py`: endpoint wiring and response assembly.
- `backend/app/inference.py`: VLM inference + CTA trigger logic (`maybe_append_sampling_guidance`).
- `backend/app/field_metrics.py`: incidence/severity/area formulas.
- `backend/app/schemas.py`: request/response contracts.
- `backend/app/config.py`: CTA text + trigger reminder keywords + guardrail messages.

---

## 2) What Triggers the Field Survey Text and Button in Chat

The text/button is not created by the model text itself.  
It is controlled by backend response flags and rendered by the frontend when those flags are true.

### Trigger flow in `/api/chat`

After generating the model response, backend calls:

- `maybe_append_sampling_guidance(question, response, field_metrics_state)` in `inference.py`

This function returns:

- updated `response` (currently unchanged),
- `sampling_guidance_appended` (bool),
- updated `field_metrics_state`,
- `field_survey_cta` (bool).

Then `main.py` maps it to:

- `field_survey_cta`
- `field_survey_cta_text` (from `FIELD_SURVEY_CTA_TEXT` in config, only when CTA is true)
- `sampling_guidance_appended`

### Exact conditions for CTA = true

CTA is shown only when **all required checks pass**:

1. Response is not off-topic guardrail text.
2. User question does not look like a field sample data submission (`looks_like_field_sample_submission`).
3. No previous field metrics already stored (`last_avg_incidence_pct is None`).
4. Either:
   - question contains survey-trigger keywords (severity, prevention, treatment, yield, whole field, hectares, acres, kanal, marla, etc.), or
   - user asks for reminder using keywords (`remind`, `again`, `sampling`, `quadrat`, `steps`, etc.).
5. If guidance was already offered before, CTA is normally suppressed unless reminder keywords are present.

If conditions pass, backend sets:

- `sampling_guidance_offered = true` in outgoing `field_metrics_state`
- `field_survey_cta = true`
- `field_survey_cta_text = "I recommend using the field severity calculation option ..."`

### Important behavior details

- If question is about yield/prevention/treatment/field spread, CTA commonly appears.
- If user already has computed field metrics, CTA is suppressed.
- If user submits sample-like text (totals/infected), CTA is suppressed.
- CTA logic is fully backend-driven; frontend only displays what backend signals.

---

## 3) How the API Service Works

## 3.1 `POST /api/chat` (image + conversational AI + CTA signal)

Request handling summary:

1. Validate model is loaded.
2. Parse conversation history into `(user, assistant)` pairs.
3. Decode base64 image.
4. Run `generate_answer(...)`:
   - Guard 1: off-topic question filtering
   - prompt assembly + history compaction
   - model generation
   - Guard 2: image-topic validation for first turn
   - cleanup/dedup
5. Run CTA decision (`maybe_append_sampling_guidance`).
6. Infer disease label from response text.
7. Return `ChatResponse` including CTA flags and updated state.

### Guardrails that affect chat output

- Off-topic question -> returns `OFF_TOPIC_QUESTION_RESPONSE`
- Non-rice/non-plant-like image (on first turn) -> returns `OFF_TOPIC_RESPONSE`
- In these off-topic cases, CTA is not emitted.

## 3.2 `POST /api/field-metrics/calculate` (deterministic calculator)

Request handling summary:

1. Accept structured `FieldSamplingReport`.
2. Validate schema constraints.
3. Compute metrics via `compute_field_metrics(report)`.
4. Build `FieldMetricsState` from result:
   - average incidence
   - number of samples
   - formatted estimated infected area string
   - average severity (if ratings provided)
5. Return both:
   - `field_metrics` (calculated values + narrative)
   - `field_metrics_state` (for roundtrip in future chat calls)

This endpoint is independent from model inference and can be called even when model is still loading.

---

## 4) Payloads (Request Contracts)

## 4.1 `/api/chat` request payload

Schema: `ChatRequest`

- `image` (string, required): base64 image (can include data URL prefix).
- `question` (string, required): user query.
- `conversation_history` (optional): list of `{ role, content }`.
- `max_new_tokens` (optional int): 1..96 (server hard-capped).
- `field_metrics_state` (optional object): previously returned state.

Example:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSk...",
  "question": "What yield loss should I expect in the whole field?",
  "conversation_history": [
    { "role": "user", "content": "What disease is this?" },
    { "role": "assistant", "content": "This looks like rice blast." }
  ],
  "max_new_tokens": 56,
  "field_metrics_state": {
    "sampling_guidance_offered": false,
    "last_avg_incidence_pct": null,
    "last_num_samples": null,
    "last_estimated_infected_area_display": null,
    "last_avg_severity_pct": null
  }
}
```

## 4.2 `/api/field-metrics/calculate` request payload

Schema: `FieldSamplingReport`

- `total_field_area` (float > 0)
- `area_unit` (`m2` | `kanal` | `marla` | `acre` | `ha`)
- `quadrat_radius_m` (optional float > 0)
- `samples` (array, min length 1), each sample has:
  - `quadrant_id` (optional 1..4)
  - `total_plants` (int > 0)
  - `infected_plants` (int >= 0 and <= total_plants)
  - `rating_counts` (optional array of `{ rating: 0..5, count >= 0 }`, and sum(count) must equal `total_plants`)

Example:

```json
{
  "total_field_area": 10,
  "area_unit": "ha",
  "quadrat_radius_m": 1,
  "samples": [
    {
      "quadrant_id": 1,
      "total_plants": 20,
      "infected_plants": 8,
      "rating_counts": [
        { "rating": 0, "count": 6 },
        { "rating": 3, "count": 8 },
        { "rating": 5, "count": 6 }
      ]
    },
    {
      "quadrant_id": 2,
      "total_plants": 20,
      "infected_plants": 10
    }
  ]
}
```

---

## 5) Formula Used for Calculation

Formulas are in `backend/app/field_metrics.py`.

## 5.1 Per-sample incidence (%)

For each sample:

`incidence_pct = (infected_plants / total_plants) * 100`

Rounded to 2 decimals.

## 5.2 Average incidence (%)

`average_incidence_pct = mean(per_sample_incidence_pct)`

Rounded to 2 decimals.

## 5.3 Estimated infected area (same unit as input field area)

`estimated_infected_area = (average_incidence_pct / 100) * total_field_area`

Rounded to 4 decimals.

No unit conversion is applied in this formula; output unit is exactly the provided `area_unit`.

## 5.4 Quadrat area (optional)

If `quadrat_radius_m` is provided:

`quadrat_area_m2 = pi * r^2`

Rounded to 4 decimals.

## 5.5 Severity index (%) from 0-5 ratings (optional)

For each sample with `rating_counts`:

1. Weighted sum:
   - `w = sum(rating * count)`
2. Severity:
   - `sample_severity_pct = (w / (MAX_RATING * total_plants)) * 100`
   - where `MAX_RATING = 5`

Rounded to 2 decimals.

Average severity:

- mean of only the rated samples (`rated_only`), rounded to 2 decimals.
- unrated samples get `0.0` in per-sample severity output list but are excluded from the average-severity denominator.

---

## 6) API Responses

## 6.1 `/api/chat` response (`ChatResponse`)

Important fields:

- `response`: model-generated text.
- `disease_detected`: `blast` | `blight` | `brownspot` | `null`.
- `context_status`: `ok` | `compacted` | `near_limit`.
- `image_rejected`: true when image guardrail rejects first-turn image.
- `field_metrics`: currently deprecated for chat path (`null`).
- `field_metrics_state`: updated state object for next turn.
- `sampling_guidance_appended`: true when field survey CTA should be shown.
- `field_survey_cta`: true when frontend should render field survey line + button.
- `field_survey_cta_text`: populated when CTA is true.

Example where CTA is triggered:

```json
{
  "response": "Rice blast can reduce yield depending on spread and severity.",
  "disease_detected": "blast",
  "context_status": "ok",
  "image_rejected": false,
  "field_metrics": null,
  "field_metrics_state": {
    "sampling_guidance_offered": true,
    "last_avg_incidence_pct": null,
    "last_num_samples": null,
    "last_estimated_infected_area_display": null,
    "last_avg_severity_pct": null
  },
  "sampling_guidance_appended": true,
  "field_survey_cta": true,
  "field_survey_cta_text": "I recommend using the field severity calculation option to know the exact severity of the field and I'll be able to give you better guidance."
}
```

## 6.2 `/api/field-metrics/calculate` response (`FieldMetricsCalculateResponse`)

Contains:

1. `field_metrics`:
   - `per_sample_incidence_pct`
   - `average_incidence_pct`
   - `num_samples`
   - `estimated_infected_area`
   - `estimated_infected_area_unit`
   - `total_field_area`
   - `quadrat_area_m2` (optional)
   - `per_sample_severity_pct` (optional)
   - `average_severity_pct` (optional)
   - `narrative` (human-readable summary)
2. `field_metrics_state`:
   - compact summary values to send back in future `/api/chat` calls.

Example:

```json
{
  "field_metrics": {
    "per_sample_incidence_pct": [40.0, 50.0],
    "average_incidence_pct": 45.0,
    "num_samples": 2,
    "estimated_infected_area": 4.5,
    "estimated_infected_area_unit": "ha",
    "total_field_area": 10.0,
    "quadrat_area_m2": 3.1416,
    "per_sample_severity_pct": [44.0, 0.0],
    "average_severity_pct": 44.0,
    "narrative": "Based on your 2 samples, here is what the numbers tell us about your field..."
  },
  "field_metrics_state": {
    "sampling_guidance_offered": false,
    "last_avg_incidence_pct": 45.0,
    "last_num_samples": 2,
    "last_estimated_infected_area_display": "4.50 hectares",
    "last_avg_severity_pct": 44.0
  }
}
```

---

## 7) Notes and Integration Implications

1. Field sample free-text parser (`field_sampling_parse.py`) is currently used to detect sample-like user messages and suppress CTA, but calculation endpoint expects **structured JSON**.
2. CTA is an API contract signal (`field_survey_cta`, `field_survey_cta_text`) and should be treated as UI instruction by frontend.
3. Once real field metrics are present in state (`last_avg_incidence_pct` set), backend stops re-suggesting CTA for normal trigger keywords.
4. Reminder keywords can re-enable CTA prompt behavior even after the first offer, but still not after metrics are already computed.

