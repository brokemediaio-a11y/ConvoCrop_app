'use client'

import { useState, KeyboardEvent } from 'react'
import GlassSurface from './GlassSurface'

interface ChatInputProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
}

export default function ChatInput({ onSendMessage, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('')

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim())
      setMessage('')
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex gap-2">
      <div className="flex-1 relative">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about the rice disease..."
          disabled={disabled}
          rows={1}
          className="w-full px-4 py-2 bg-[#1a1a1a] border border-[#00a71b]/30 rounded-lg text-[#dedede] placeholder-[#dedede]/50 focus:outline-none focus:ring-2 focus:ring-[#00a71b] focus:border-[#00a71b] resize-none disabled:opacity-50 disabled:cursor-not-allowed"
          style={{ minHeight: '44px', maxHeight: '120px' }}
        />
      </div>
      <GlassSurface
        width={100}
        height={44}
        borderRadius={22}
        onClick={disabled || !message.trim() ? undefined : handleSend}
        className={`${(disabled || !message.trim()) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <div className="flex items-center gap-2">
          <span className="text-[#dedede] font-medium">Send</span>
          <svg
            className="w-5 h-5 text-[#dedede]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </div>
      </GlassSurface>
    </div>
  )
}
