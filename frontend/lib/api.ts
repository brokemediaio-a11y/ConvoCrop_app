import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  /** Assistant only: show field survey line + button inside the bubble */
  fieldSurveyCta?: boolean
  fieldSurveyCtaText?: string
  imageQualityIssue?: string
}

export type AreaUnit = 'm2' | 'kanal' | 'marla' | 'acre' | 'ha'

export function fieldMetricsStorageKey(sessionId: string): string {
  return `convocrop_field_metrics_${sessionId}`
}

/** Persisted when navigating to /field-severity so /chat can restore the same session */
export const CHAT_SESSION_SNAPSHOT_KEY = 'convocrop_chat_snapshot_v1'

export interface RatingCount {
  rating: number
  count: number
}

export interface FieldSample {
  quadrant_id?: number
  total_plants: number
  infected_plants: number
  rating_counts?: RatingCount[]
}

export interface FieldSamplingReport {
  total_field_area: number
  area_unit: AreaUnit
  quadrat_radius_m?: number
  samples: FieldSample[]
}

export interface FieldMetricsState {
  sampling_guidance_offered?: boolean
  last_avg_incidence_pct?: number
  last_num_samples?: number
  last_estimated_infected_area_display?: string
  last_avg_severity_pct?: number
  last_estimated_infected_area_ha?: number
  last_field_area_ha?: number
  field_severity_tier?: 'healthy' | 'mild' | 'moderate' | 'severe'
  last_detected_disease?: 'blast' | 'blight' | 'brownspot'
}

export interface FieldMetricsResult {
  per_sample_incidence_pct: number[]
  average_incidence_pct: number
  num_samples: number
  estimated_infected_area: number
  estimated_infected_area_unit: AreaUnit
  total_field_area: number
  quadrat_area_m2?: number
  per_sample_severity_pct?: number[]
  average_severity_pct?: number
  narrative: string
}

export interface FieldMetricsCalculateResponse {
  field_metrics: FieldMetricsResult
  field_metrics_state: FieldMetricsState
}

export interface ChatRequest {
  image: string
  question: string
  conversation_history?: ChatMessage[]
  max_new_tokens?: number
  field_metrics_state?: FieldMetricsState
  session_id?: string
  signal?: AbortSignal
}

export interface ChatResponse {
  response: string
  disease_detected?: string
  image_rejected?: boolean
  image_quality_issue?: string | null
  context_status?: string
  field_metrics?: FieldMetricsResult
  field_metrics_state?: FieldMetricsState
  sampling_guidance_appended?: boolean
  field_survey_cta?: boolean
  field_survey_cta_text?: string | null
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  device: string
  model_path: string
}

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for CPU inference
  headers: {
    'Content-Type': 'application/json',
  },
})

const MAX_CHAT_RETRIES = 2
const RETRYABLE_STATUS = new Set([408, 429, 500, 502, 503, 504])

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function shouldRetry(error: unknown): boolean {
  if (!axios.isAxiosError(error)) return false
  if (error.code === 'ERR_CANCELED') return false
  if (error.code === 'ECONNABORTED') return true
  if (!error.response) return true
  return RETRYABLE_STATUS.has(error.response.status)
}

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/api/health')
  return response.data
}

export const sendChatMessage = async (
  request: ChatRequest
): Promise<ChatResponse> => {
  const { signal, session_id, ...rest } = request
  const payload: ChatRequest = {
    ...rest,
    conversation_history: rest.conversation_history?.map(({ role, content }) => ({
      role,
      content,
    })),
  }
  let lastError: unknown = null
  for (let attempt = 0; attempt <= MAX_CHAT_RETRIES; attempt += 1) {
    try {
      const response = await api.post<ChatResponse>('/api/chat', payload, {
        signal,
        headers: session_id ? { 'X-Chat-Session-Id': session_id } : undefined,
      })
      return response.data
    } catch (error) {
      lastError = error
      if (!shouldRetry(error) || attempt === MAX_CHAT_RETRIES) {
        throw error
      }
      await sleep(400 * (attempt + 1))
    }
  }
  throw lastError
}

export const calculateFieldMetrics = async (
  report: FieldSamplingReport
): Promise<FieldMetricsCalculateResponse> => {
  const response = await api.post<FieldMetricsCalculateResponse>(
    '/api/field-metrics/calculate',
    report
  )
  return response.data
}

export const uploadImageAndChat = async (
  file: File,
  question: string,
  maxNewTokens: number = 200
): Promise<ChatResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('question', question)
  formData.append('max_new_tokens', maxNewTokens.toString())

  const response = await api.post<ChatResponse>('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}
