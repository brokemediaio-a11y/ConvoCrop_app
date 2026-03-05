import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  image: string
  question: string
  conversation_history?: ChatMessage[]
  max_new_tokens?: number  // Default: 100 for concise answers
}

export interface ChatResponse {
  response: string
  disease_detected?: string
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

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/api/health')
  return response.data
}

export const sendChatMessage = async (
  request: ChatRequest
): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>('/api/chat', request)
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
