'use client'

import { useState, useRef, useEffect } from 'react'
import ImageUpload from './ImageUpload'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import GlassSurface from './GlassSurface'
import { sendChatMessage, ChatMessage } from '@/lib/api'

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleImageSelect = (imageBase64: string) => {
    setCurrentImage(imageBase64)
    setError(null)
  }

  const handleSendMessage = async (question: string) => {
    if (!currentImage) {
      setError('Please upload an image first')
      return
    }

    if (!question.trim()) {
      return
    }

    // Add user message
    const userMessage: ChatMessage = {
      role: 'user',
      content: question,
    }
    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)
    setError(null)

    try {
      // Prepare conversation history (excluding the current question)
      const conversationHistory: ChatMessage[] = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }))

      const response = await sendChatMessage({
        image: currentImage,
        question: question,
        conversation_history: conversationHistory,
        max_new_tokens: 100,
      })

      // Add assistant response
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err: any) {
      console.error('Error sending message:', err)
      setError(
        err.response?.data?.detail || err.message || 'Failed to get response'
      )
      // Remove the user message on error
      setMessages((prev) => prev.slice(0, -1))
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearChat = () => {
    setMessages([])
    setError(null)
  }

  const handleNewImage = () => {
    setCurrentImage(null)
    setMessages([])
    setError(null)
  }

  return (
    <div className="rounded-lg overflow-hidden flex flex-col h-[calc(100vh-250px)] max-h-[800px] border border-[#00a71b]/20">
      {/* Header */}
      <div className="bg-[#212121] p-4 flex justify-between items-center border-b border-[#00a71b]/20">
        <h2 className="text-xl font-extrabold text-[#dedede]">Chat with AI Assistant</h2>
        <div className="flex gap-2">
          {currentImage && (
            <GlassSurface
              width={110}
              height={36}
              borderRadius={18}
              onClick={handleNewImage}
              className="cursor-pointer"
            >
              <span className="text-[#dedede] font-medium text-sm">New Image</span>
            </GlassSurface>
          )}
          {messages.length > 0 && (
            <GlassSurface
              width={110}
              height={36}
              borderRadius={18}
              onClick={handleClearChat}
              className="cursor-pointer"
            >
              <span className="text-[#dedede] font-medium text-sm">Clear Chat</span>
            </GlassSurface>
          )}
        </div>
      </div>

      {/* Image Upload Section */}
      {!currentImage && (
        <div className="p-6 border-b border-[#00a71b]/20 bg-[#212121]">
          <ImageUpload onImageSelect={handleImageSelect} />
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-900/30 border-l-4 border-red-500 text-red-300 p-4 mx-4 mt-4 rounded">
          <p className="font-medium">Error</p>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 bg-[#212121]">
        {currentImage && (
          <div className="mb-4">
            <div className="inline-block max-w-xs">
              <img
                src={`data:image/jpeg;base64,${currentImage}`}
                alt="Uploaded rice leaf"
                className="rounded-lg shadow-md max-w-full h-auto border-2 border-[#00a71b]/30"
              />
            </div>
          </div>
        )}

        {messages.length === 0 && currentImage && (
          <div className="text-center text-[#dedede] opacity-70 py-8">
            <p className="font-medium">Ask me about the rice disease in the image!</p>
            <p className="text-sm mt-2">Try: &quot;What disease does this rice leaf have?&quot;</p>
          </div>
        )}

        <MessageList messages={messages} />

        {isLoading && (
          <div className="flex items-center gap-2 text-[#dedede] opacity-70 py-4">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-[#00a71b]"></div>
            <span className="font-medium">AI is thinking...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Section */}
      {currentImage && (
        <div className="border-t border-[#00a71b]/20 p-4 bg-[#212121]">
          <ChatInput
            onSendMessage={handleSendMessage}
            disabled={isLoading}
          />
        </div>
      )}
    </div>
  )
}