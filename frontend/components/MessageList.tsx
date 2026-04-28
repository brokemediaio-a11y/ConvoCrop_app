'use client'

import Image from 'next/image'
import { ChatMessage } from '@/lib/api'

interface MessageListProps {
  messages: ChatMessage[]
  onOpenFieldSurvey?: () => void
  suggestedPrompts?: string[]
  onSuggestionClick?: (suggestion: string) => void
}

export default function MessageList({
  messages,
  onOpenFieldSurvey,
  suggestedPrompts = [],
  onSuggestionClick,
}: MessageListProps) {
  if (messages.length === 0) {
    return null
  }

  const lastAssistantIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant') return i
    }
    return -1
  })()

  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-[80%] md:max-w-[70%] rounded-xl px-4 py-3 ${
              message.role === 'user'
                ? 'bg-[#1a1a1a] text-white'
                : 'bg-white/60 text-[#252525] border border-[#252525]/10'
            }`}
            style={message.role === 'assistant' ? { backdropFilter: 'blur(8px)' } : undefined}
          >
            <div className="flex items-start gap-2">
              {message.role === 'assistant' && (
                <div className="flex-shrink-0 mt-0.5 w-5 h-5 relative">
                  <Image
                    src="/logos/favicon.png"
                    alt=""
                    fill
                    className="object-contain"
                  />
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs font-bold mb-1 opacity-60">
                  {message.role === 'user' ? 'You' : 'Convo Crop'}
                </p>
                <p className="whitespace-pre-wrap break-words text-sm font-medium leading-relaxed">
                  {message.content}
                </p>
                {message.role === 'assistant' && message.imageQualityIssue && (
                  <div className="mt-2 inline-flex items-center rounded-full border border-amber-300/70 bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-800">
                    Why rejected: {message.imageQualityIssue}
                  </div>
                )}
                {message.role === 'assistant' && message.fieldSurveyCta && onOpenFieldSurvey && (
                  <div className="mt-3 pt-3 border-t border-[#252525]/10">
                    <p className="text-sm font-medium text-[#252525]/90 mb-2 leading-snug">
                      {message.fieldSurveyCtaText ??
                        "I recommend using the field severity calculation option to know the exact severity of the field and I'll be able to give you better guidance."}
                    </p>
                    <button
                      type="button"
                      onClick={onOpenFieldSurvey}
                      className="w-full sm:w-auto rounded-xl bg-[#c6deca] hover:bg-[#b5cfba] text-[#1e2f24] text-sm font-semibold px-4 py-2.5 transition-colors border border-[#252525]/10 shadow-sm"
                    >
                      Field Severity
                    </button>
                  </div>
                )}
                {message.role === 'assistant' &&
                  index === lastAssistantIndex &&
                  suggestedPrompts.length > 0 &&
                  onSuggestionClick && (
                    <div className="mt-3 pt-3 border-t border-[#252525]/10">
                      <p className="text-xs font-semibold uppercase tracking-wide text-[#252525]/60 mb-2">
                        Suggested next questions
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {suggestedPrompts.map((prompt) => (
                          <button
                            key={prompt}
                            type="button"
                            onClick={() => onSuggestionClick(prompt)}
                            className="w-full sm:w-auto rounded-full border border-[#252525]/15 bg-[#f5f7f5] hover:bg-[#ecf2ec] text-[#252525] text-xs sm:text-sm font-medium px-3 py-2 text-left transition-colors"
                          >
                            {prompt}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
