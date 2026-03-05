'use client'

import { useRouter } from 'next/navigation'
import ChatInterface from '@/components/ChatInterface'
import GlassSurface from '@/components/GlassSurface'

export default function ChatPage() {
  const router = useRouter()

  return (
    <div className="min-h-screen bg-[#212121] relative">
      {/* Header with Back Button */}
      <div className="absolute top-6 left-6 z-20">
        <GlassSurface
          width={120}
          height={45}
          borderRadius={20}
          onClick={() => router.push('/')}
          className="cursor-pointer hover:opacity-90 transition-opacity"
        >
          <span className="text-[#dedede] font-medium">← Back</span>
        </GlassSurface>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8 pt-20">
        <div className="max-w-5xl mx-auto">
          <header className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-extrabold text-[#dedede] mb-2">
              Convo Crop
            </h1>
            <p className="text-lg font-medium text-[#dedede] opacity-80">
              AI-powered chatbot for identifying rice plant diseases
            </p>
          </header>
          <ChatInterface />
        </div>
      </div>
    </div>
  )
}
