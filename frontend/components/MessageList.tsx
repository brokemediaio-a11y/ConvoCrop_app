'use client'

import { ChatMessage } from '@/lib/api'

interface MessageListProps {
  messages: ChatMessage[]
}

export default function MessageList({ messages }: MessageListProps) {
  if (messages.length === 0) {
    return null
  }

  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`flex ${
            message.role === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          <div
            className={`max-w-[80%] md:max-w-[70%] rounded-lg px-4 py-3 ${
              message.role === 'user'
                ? 'bg-[#00a71b] text-[#dedede]'
                : 'bg-[#1a1a1a] text-[#dedede] border border-[#00a71b]/30'
            }`}
          >
            <div className="flex items-start gap-2">
              {message.role === 'assistant' && (
                <div className="flex-shrink-0 mt-1">
                  <svg
                    className="w-5 h-5 text-[#00a71b]"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
              )}
              <div className="flex-1">
                <p className="text-sm font-extrabold mb-1">
                  {message.role === 'user' ? 'You' : 'AI Assistant'}
                </p>
                <p className="whitespace-pre-wrap break-words font-medium">
                  {message.content}
                </p>
                {message.role === 'assistant' && (
                  <div className="mt-2 text-xs text-[#dedede] opacity-50">
                    {new Date().toLocaleTimeString()}
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
