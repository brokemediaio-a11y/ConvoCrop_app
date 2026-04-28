'use client'

import { useState, useRef, useEffect, useLayoutEffect, useCallback, Suspense, useMemo } from 'react'
import Image from 'next/image'
import { useRouter, usePathname, useSearchParams } from 'next/navigation'
import GlassSurface from '@/components/GlassSurface'
import ImageUpload from '@/components/ImageUpload'
import MessageList from '@/components/MessageList'
import {
  sendChatMessage,
  ChatMessage,
  FieldMetricsState,
  fieldMetricsStorageKey,
  CHAT_SESSION_SNAPSHOT_KEY,
} from '@/lib/api'
import { getNextPromptSuggestions } from '@/lib/nextPromptSuggestions'
import { CONVOCROP_LOGO_SRC } from '@/lib/assets'

interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  image: string | null
  imageRejected: boolean
  fieldMetricsState?: FieldMetricsState | null
}

interface ChatSnapshot {
  sessions: ChatSession[]
  activeSessionId: string | null
}

function ChatLoadingFallback() {
  return (
    <div className="h-screen w-screen flex items-center justify-center bg-[#ebebeb]">
      <span className="text-[#252525]/60 text-sm">Loading...</span>
    </div>
  )
}

function ChatPageInner() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chatDisabled, setChatDisabled] = useState(false)
  const [loadingText, setLoadingText] = useState('Analyzing image...')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const loadingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const sessionsRef = useRef<ChatSession[]>([])
  const restoredFromFieldSurveyRef = useRef(false)
  const activeRequestControllerRef = useRef<AbortController | null>(null)
  const requestSeqRef = useRef(0)

  useEffect(() => {
    sessionsRef.current = sessions
  }, [sessions])

  useEffect(() => {
    return () => {
      activeRequestControllerRef.current?.abort()
      activeRequestControllerRef.current = null
    }
  }, [])

  const mergeFieldMetricsForSession = useCallback((sid: string, state: FieldMetricsState) => {
    if (!sessionsRef.current.some((s) => s.id === sid)) return
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sid
          ? { ...s, fieldMetricsState: { ...(s.fieldMetricsState ?? {}), ...state } }
          : s,
      ),
    )
    setActiveSessionId(sid)
  }, [])

  useLayoutEffect(() => {
    const sid = searchParams.get('sessionId')
    const flag = searchParams.get('fieldMetrics')
    if (flag !== '1' || !sid || restoredFromFieldSurveyRef.current) return
    restoredFromFieldSurveyRef.current = true

    let snapshot: ChatSnapshot | null = null
    try {
      const raw = sessionStorage.getItem(CHAT_SESSION_SNAPSHOT_KEY)
      if (raw) snapshot = JSON.parse(raw) as ChatSnapshot
    } catch {
      snapshot = null
    }

    let fieldSt: FieldMetricsState | null = null
    try {
      const mr = sessionStorage.getItem(fieldMetricsStorageKey(sid))
      if (mr) fieldSt = JSON.parse(mr) as FieldMetricsState
    } catch {
      fieldSt = null
    }

    if (snapshot?.sessions?.length) {
      let nextSessions = snapshot.sessions
      if (fieldSt) {
        nextSessions = nextSessions.map((s) =>
          s.id === sid
            ? { ...s, fieldMetricsState: { ...(s.fieldMetricsState ?? {}), ...fieldSt } }
            : s,
        )
      }
      const aid = snapshot.activeSessionId || sid
      setSessions(nextSessions)
      setActiveSessionId(aid)
      const cur = nextSessions.find((s) => s.id === aid)
      if (cur) {
        setMessages(cur.messages)
        setCurrentImage(cur.image)
        setChatDisabled(cur.imageRejected)
      }
    } else if (fieldSt) {
      mergeFieldMetricsForSession(sid, fieldSt)
    }

    try {
      sessionStorage.removeItem(CHAT_SESSION_SNAPSHOT_KEY)
    } catch {
      /* ignore */
    }

    router.replace(pathname)
  }, [searchParams, pathname, router, mergeFieldMetricsForSession])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const LOADING_PHRASES = [
    'Analyzing image...',
    'Classifying disease...',
    'Detecting severity...',
    'Identifying affected regions...',
    'Generating diagnosis...',
  ]

  useEffect(() => {
    if (isLoading) {
      let idx = 0
      setLoadingText(LOADING_PHRASES[0])
      loadingIntervalRef.current = setInterval(() => {
        idx = (idx + 1) % LOADING_PHRASES.length
        setLoadingText(LOADING_PHRASES[idx])
      }, 2200)
    } else {
      if (loadingIntervalRef.current) {
        clearInterval(loadingIntervalRef.current)
        loadingIntervalRef.current = null
      }
    }
    return () => {
      if (loadingIntervalRef.current) {
        clearInterval(loadingIntervalRef.current)
      }
    }
  }, [isLoading])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`
    }
  }, [input])

  const activeSession = sessions.find((s) => s.id === activeSessionId)
  const suggestedPrompts = useMemo(() => {
    if (isLoading || chatDisabled) return []
    return getNextPromptSuggestions(messages, 2)
  }, [messages, isLoading, chatDisabled])

  function createNewChat() {
    activeRequestControllerRef.current?.abort()
    activeRequestControllerRef.current = null
    requestSeqRef.current += 1
    const id = crypto.randomUUID()
    const session: ChatSession = {
      id,
      title: 'New Chat',
      messages: [],
      image: null,
      imageRejected: false,
      fieldMetricsState: null,
    }
    setSessions((prev) => [session, ...prev])
    setActiveSessionId(id)
    setMessages([])
    setCurrentImage(null)
    setError(null)
    setChatDisabled(false)
    setInput('')
  }

  function switchSession(id: string) {
    activeRequestControllerRef.current?.abort()
    activeRequestControllerRef.current = null
    requestSeqRef.current += 1
    const session = sessions.find((s) => s.id === id)
    if (!session) return
    setActiveSessionId(id)
    setMessages(session.messages)
    setCurrentImage(session.image)
    setChatDisabled(session.imageRejected)
    setError(null)
    setInput('')
  }

  function handleImageSelect(imageBase64: string) {
    activeRequestControllerRef.current?.abort()
    activeRequestControllerRef.current = null
    requestSeqRef.current += 1
    setCurrentImage(imageBase64)
    setError(null)
    if (activeSessionId) {
      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, image: imageBase64 } : s)),
      )
    }
  }

  async function handleSend() {
    if (!input.trim()) return
    if (isLoading) {
      setError('A request is already in progress. Please wait for the current response.')
      return
    }

    if (!currentImage) {
      setError('Please upload an image first')
      return
    }

    const userMsg: ChatMessage = { role: 'user', content: input.trim() }
    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    setInput('')
    setIsLoading(true)
    setError(null)

    if (activeSessionId) {
      const title = messages.length === 0 ? input.trim().slice(0, 40) : undefined
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? { ...s, messages: updatedMessages, ...(title ? { title } : {}) }
            : s,
        ),
      )
    }

    const requestId = requestSeqRef.current + 1
    requestSeqRef.current = requestId
    const controller = new AbortController()
    activeRequestControllerRef.current = controller

    try {
      const history: ChatMessage[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }))
      const fieldMetricsState = activeSession?.fieldMetricsState ?? undefined
      const response = await sendChatMessage({
        image: currentImage,
        question: input.trim(),
        conversation_history: history,
        max_new_tokens: 64,
        field_metrics_state: fieldMetricsState,
        session_id: activeSessionId ?? undefined,
        signal: controller.signal,
      })

      if (requestId !== requestSeqRef.current) {
        return
      }

      const aiMsg: ChatMessage = {
        role: 'assistant',
        content: response.response,
        fieldSurveyCta: response.field_survey_cta === true,
        fieldSurveyCtaText: response.field_survey_cta_text ?? undefined,
        imageQualityIssue: response.image_quality_issue ?? undefined,
      }
      const finalMessages = [...updatedMessages, aiMsg]
      setMessages(finalMessages)

      const rejected = response.image_rejected === true
      if (rejected) {
        setChatDisabled(true)
      }

      if (activeSessionId) {
        setSessions((prev) =>
          prev.map((s) =>
            s.id === activeSessionId
              ? {
                  ...s,
                  messages: finalMessages,
                  ...(rejected ? { imageRejected: true } : {}),
                  fieldMetricsState: response.field_metrics_state
                    ? { ...(s.fieldMetricsState ?? {}), ...response.field_metrics_state }
                    : s.fieldMetricsState,
                }
              : s,
          ),
        )
      }
    } catch (err: unknown) {
      if (controller.signal.aborted || requestId !== requestSeqRef.current) {
        return
      }
      const msg =
        err && typeof err === 'object' && 'response' in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null
      const fallback =
        err && typeof err === 'object' && 'code' in err && (err as { code?: string }).code === 'ECONNABORTED'
          ? 'The request timed out on this device. Please try again with a clearer image or fewer follow-up turns.'
          : 'Connection dropped during inference. Please try again.'
      setError(typeof msg === 'string' ? msg : fallback)
      setMessages((prev) => prev.slice(0, -1))
    } finally {
      if (activeRequestControllerRef.current === controller) {
        activeRequestControllerRef.current = null
      }
      setIsLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const showUpload = !currentImage

  function navigateToFieldSurvey() {
    if (!activeSessionId) return
    restoredFromFieldSurveyRef.current = false
    try {
      sessionStorage.setItem(
        CHAT_SESSION_SNAPSHOT_KEY,
        JSON.stringify({ sessions, activeSessionId }),
      )
    } catch {
      /* quota or private mode */
    }
    router.push(`/field-severity?sessionId=${encodeURIComponent(activeSessionId)}`)
  }

  return (
    <div className="h-screen w-screen overflow-hidden relative flex">
      {/* Background */}
      <div className="fixed inset-0 z-0 bg-[#ebebeb]" />

      {/* Sidebar */}
      <aside
        className={`relative z-20 flex flex-col h-full transition-all duration-300 ${
          sidebarOpen ? 'w-64' : 'w-0'
        } overflow-hidden shrink-0`}
      >
        <div
          className="flex flex-col h-full w-64 border-r border-[#252525]/10"
          style={{ background: 'rgba(235,235,235,0.6)', backdropFilter: 'blur(16px)' }}
        >
          {/* Logo */}
          <div className="px-4 pt-5 pb-3 flex items-center gap-2">
            <div className="relative w-32 h-8 cursor-pointer" onClick={() => router.push('/')}>
              <Image
                src={CONVOCROP_LOGO_SRC}
                alt="Convo Crop"
                fill
                className="object-contain"
                priority
              />
            </div>
          </div>

          {/* New Chat Button */}
          <div className="px-3 pb-3">
            <GlassSurface
              width="100%"
              height={42}
              borderRadius={12}
              onClick={createNewChat}
              className="cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#252525]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span className="text-[#252525] font-medium text-sm">New Chat</span>
              </div>
            </GlassSurface>
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto px-3 space-y-1">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => switchSession(session.id)}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-sm truncate transition-colors ${
                  session.id === activeSessionId
                    ? 'bg-[#00a71b]/15 text-[#252525] font-medium'
                    : 'text-[#252525]/70 hover:bg-[#252525]/5'
                }`}
              >
                {session.title}
              </button>
            ))}
            {sessions.length === 0 && (
              <p className="text-xs text-[#252525]/40 text-center mt-8 px-2">
                No conversations yet. Start a new chat.
              </p>
            )}
          </div>

          {/* Back to Home */}
          <div className="px-3 py-3 border-t border-[#252525]/10">
            <button
              onClick={() => router.push('/')}
              className="flex items-center gap-2 text-sm text-[#252525]/60 hover:text-[#252525] transition-colors w-full px-2 py-1.5"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4"
                />
              </svg>
              Back to Home
            </button>
            <p className="mt-3 px-2 text-[11px] leading-relaxed text-[#252525]/55">
              AI can make mistakes. Always consult a plant pathologist before taking action.
            </p>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="relative z-10 flex-1 flex flex-col h-full min-w-0">
        {/* Top Bar */}
        <div className="flex items-center gap-3 px-4 py-3 shrink-0 flex-wrap">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-[#252525]/5 transition-colors text-[#252525]/60"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          {activeSession && (
            <h2 className="text-sm font-medium text-[#252525]/70 truncate min-w-0 flex-1">
              {activeSession.title}
            </h2>
          )}

          <div className="ml-auto flex items-center gap-2 flex-wrap justify-end">
            {activeSessionId && (
              <GlassSurface
                width={130}
                height={34}
                borderRadius={10}
                onClick={navigateToFieldSurvey}
                className="cursor-pointer"
              >
                <span className="text-[#252525] font-medium text-xs px-1">Field Severity</span>
              </GlassSurface>
            )}
            {currentImage && !chatDisabled && (
              <GlassSurface
                width={110}
                height={34}
                borderRadius={10}
                onClick={() => {
                  setCurrentImage(null)
                  if (activeSessionId) {
                    setSessions((prev) =>
                      prev.map((s) => (s.id === activeSessionId ? { ...s, image: null } : s)),
                    )
                  }
                }}
                className="cursor-pointer"
              >
                <span className="text-[#252525] font-medium text-xs">Change Image</span>
              </GlassSurface>
            )}
          </div>
        </div>

        {/* Chat Content */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 lg:px-16">
          {!activeSessionId ? (
            <div className="flex flex-col items-center justify-center h-full gap-6">
              <div className="relative w-48 h-12">
                <Image
                  src={CONVOCROP_LOGO_SRC}
                  alt="Convo Crop"
                  fill
                  className="object-contain"
                  priority
                />
              </div>
              <p className="text-[#252525]/50 text-sm">Start a new chat to begin</p>
              <GlassSurface
                width={180}
                height={48}
                borderRadius={14}
                onClick={createNewChat}
                className="cursor-pointer"
              >
                <div className="flex items-center gap-2">
                  <svg className="w-5 h-5 text-[#252525]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  <span className="text-[#252525] font-medium">New Chat</span>
                </div>
              </GlassSurface>
            </div>
          ) : showUpload ? (
            <div className="flex flex-col items-center justify-center h-full max-w-lg mx-auto gap-4">
              <h3 className="text-xl font-bold text-[#252525] mb-2">Upload a Plant Image</h3>
              <p className="text-sm text-[#252525]/60 text-center mb-4">
                Share an image of a crop leaf and ask questions about potential diseases.
              </p>
              <ImageUpload onImageSelect={handleImageSelect} />
            </div>
          ) : (
            <div className="max-w-3xl mx-auto py-4">
              {/* Uploaded Image Thumbnail */}
              <div className="mb-4 flex items-center gap-3">
                <img
                  src={`data:image/jpeg;base64,${currentImage}`}
                  alt="Uploaded"
                  className="w-16 h-16 rounded-lg object-cover border-2 border-[#00a71b]/30"
                />
                <span className="text-xs text-[#252525]/50">Image uploaded. Ask a question below.</span>
              </div>

              {error && (
                <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-3 mb-4 rounded text-sm">
                  {error}
                </div>
              )}

              <MessageList
                messages={messages}
                onOpenFieldSurvey={navigateToFieldSurvey}
                suggestedPrompts={suggestedPrompts}
                onSuggestionClick={(prompt) => {
                  setInput(prompt)
                  textareaRef.current?.focus()
                }}
              />

              {isLoading && (
                <div className="flex items-center gap-3 py-4">
                  <div className="w-6 h-6 relative animate-spin" style={{ animationDuration: '1.5s' }}>
                    <Image src="/logos/favicon.png" alt="" fill className="object-contain" />
                  </div>
                  <span className="text-sm font-medium text-[#252525]/60 transition-opacity">
                    {loadingText}
                  </span>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar or Disabled Prompt */}
        {activeSessionId && currentImage && (
          <div className="shrink-0 px-4 md:px-8 lg:px-16 pb-4 pt-2">
            <div className="max-w-3xl mx-auto">
              {chatDisabled ? (
                <div className="flex flex-col items-center gap-3 py-2">
                  <p className="text-sm text-[#252525]/50 text-center">
                    This image does not appear to be a rice leaf. Please start a new chat with a valid plant image.
                  </p>
                  <GlassSurface
                    width={180}
                    height={46}
                    borderRadius={14}
                    onClick={createNewChat}
                    className="cursor-pointer"
                  >
                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4 text-[#252525]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                      </svg>
                      <span className="text-[#252525] font-medium text-sm">Start New Chat</span>
                    </div>
                  </GlassSurface>
                </div>
              ) : (
                <GlassSurface
                  width="100%"
                  height={56}
                  borderRadius={16}
                  blur={20}
                  opacity={0.95}
                  backgroundOpacity={0.08}
                  className="overflow-hidden"
                >
                  <div className="flex items-center w-full h-full px-4 gap-3">
                    <textarea
                      ref={textareaRef}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask about this plant..."
                      disabled={isLoading}
                      rows={1}
                      className="flex-1 bg-transparent text-[#252525] placeholder-[#252525]/40 text-sm font-medium focus:outline-none resize-none disabled:opacity-50 leading-normal"
                      style={{ maxHeight: '80px' }}
                    />
                    <button
                      onClick={handleSend}
                      disabled={isLoading || !input.trim()}
                      className="shrink-0 w-9 h-9 flex items-center justify-center rounded-full bg-[#00a71b] hover:bg-[#00a71b]/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 12h14M12 5l7 7-7 7" />
                      </svg>
                    </button>
                  </div>
                </GlassSurface>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default function ChatPage() {
  return (
    <Suspense fallback={<ChatLoadingFallback />}>
      <ChatPageInner />
    </Suspense>
  )
}
