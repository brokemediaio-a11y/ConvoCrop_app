'use client'

import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  hasImage?: boolean
}

const chatSequence: ChatMessage[] = [
  {
    role: 'user',
    content: 'What disease do you see in this image',
    hasImage: true,
  },
  {
    role: 'assistant',
    content: 'Based on the pattern, this is rice blast.',
  },
  {
    role: 'user',
    content: 'how can i prevent the further spread of this disease?',
  },
  {
    role: 'assistant',
    content: 'Stop nitrogen and keep the field dry. Check every seven days.',
  },
  {
    role: 'user',
    content: 'what factors cause this disease?',
  },
  {
    role: 'assistant',
    content: 'High humidity and long leaves give it time to spread. It needs 8+ hours of wetness.',
  },
]

export default function AnimatedChat() {
  const containerRef = useRef<HTMLDivElement>(null)
  const messagesRef = useRef<HTMLDivElement>(null)
  const [currentMessages, setCurrentMessages] = useState<ChatMessage[]>([])
  const [animationKey, setAnimationKey] = useState(0)
  const animationRef = useRef<gsap.core.Timeline | null>(null)
  const messageRefs = useRef<(HTMLDivElement | null)[]>([])
  const currentIndexRef = useRef(0)

  // Animate messages as they're added
  useEffect(() => {
    if (currentMessages.length === 0) return

    const lastIndex = currentMessages.length - 1
    const message = currentMessages[lastIndex]
    
    if (!message) return

    // Wait for DOM to be ready with retry mechanism
    const animateMessage = (retries = 0) => {
      const messageEl = messageRefs.current[lastIndex]
      
      if (!messageEl && retries < 10) {
        // Retry if element not found yet
        setTimeout(() => animateMessage(retries + 1), 50)
        return
      }
      
      if (!messageEl) return

      const isUser = message.role === 'user'
      const slideDistance = 30

      // Set initial state
      gsap.set(messageEl, {
        opacity: 0,
        x: isUser ? slideDistance : -slideDistance,
        y: 10,
      })

      // Animate in
      const tl = gsap.timeline({
        onComplete: () => {
          // Auto-scroll to bottom after animation
          if (messagesRef.current) {
            messagesRef.current.scrollTop = messagesRef.current.scrollHeight
          }
        },
      })
      tl.to(messageEl, {
        opacity: 1,
        x: 0,
        y: 0,
        duration: 0.7,
        ease: 'power3.out',
      })

      // If it's the image message, animate the image separately
      if (message.hasImage) {
        // Wait a bit for image to load
        setTimeout(() => {
          const imageEl = messageEl.querySelector('.chat-image')
          if (imageEl) {
            gsap.set(imageEl, { opacity: 0, scale: 0.95 })
            tl.to(
              imageEl,
              {
                opacity: 1,
                scale: 1,
                duration: 0.5,
                ease: 'power2.out',
              },
              '-=0.3'
            )
          }
        }, 100)
      }
    }

    // Use requestAnimationFrame and then retry if needed
    requestAnimationFrame(() => {
      animateMessage()
    })
  }, [currentMessages])

  // Main animation loop
  useEffect(() => {
    if (!messagesRef.current) return

    let timeoutId: NodeJS.Timeout
    let isRunning = true

    const addNextMessage = () => {
      if (!isRunning) return
      
      const currentIndex = currentIndexRef.current
      
      if (currentIndex < chatSequence.length) {
        // Add the message to state
        setCurrentMessages((prev) => [...prev, chatSequence[currentIndex]])
        currentIndexRef.current = currentIndex + 1

        // Schedule next message with appropriate delays
        // After image message (index 0), wait longer for image to load and animate
        // After AI responses, wait a bit longer
        let delay = 1200
        const nextIndex = currentIndexRef.current
        if (currentIndex === 0) {
          // After first message (image), wait longer for image to render and animate
          delay = 1800
        } else if (currentIndex === 1) {
          // After AI response to image, wait a bit
          delay = 1500
        } else if (chatSequence[currentIndex]?.role === 'assistant') {
          // After any AI response
          delay = 1400
        }
        
        timeoutId = setTimeout(addNextMessage, delay)
      } else {
        // All messages shown, wait then restart
        timeoutId = setTimeout(() => {
          // Clear messages and refs completely
          setCurrentMessages([])
          messageRefs.current = []
          currentIndexRef.current = 0
          // Force re-render by changing key
          setAnimationKey((prev) => prev + 1)
          // Wait longer to ensure state is fully cleared and React has re-rendered
          setTimeout(() => {
            if (isRunning) {
              // Ensure we start from index 0
              currentIndexRef.current = 0
              addNextMessage()
            }
          }, 800)
        }, 3000)
      }
    }

    // Start animation after initial delay
    const startTimeout = setTimeout(() => {
      currentIndexRef.current = 0
      addNextMessage()
    }, 800)

    return () => {
      isRunning = false
      clearTimeout(startTimeout)
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className="w-full h-full flex items-center justify-center p-4"
    >
      <div
        ref={messagesRef}
        className="w-full max-w-md h-[500px] overflow-y-auto overflow-x-hidden flex flex-col justify-end space-y-4 px-2 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
      >
        {currentMessages.map((message, index) => {
          if (!message || !message.role) return null
          
          const isUser = message.role === 'user'
          
          return (
            <div
              key={`msg-${animationKey}-${index}-${message.role}`}
              ref={(el) => {
                if (el) {
                  messageRefs.current[index] = el
                } else {
                  messageRefs.current[index] = null
                }
              }}
              className={`flex ${
                isUser ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-4 py-3 ${
                  isUser
                    ? 'bg-[#00a71b] text-[#dedede]'
                    : 'bg-[#1a1a1a] text-[#dedede] border border-[#00a71b]/30'
                }`}
              >
                <div className="flex items-start gap-2">
                  {!isUser && (
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
                    {isUser ? 'You' : 'AI Assistant'}
                  </p>
                  {message.hasImage && (
                    <div className="mb-2 chat-image">
                      <div className="relative w-48 h-32 rounded-lg overflow-hidden border-2 border-[#00a71b]/30 bg-[#1a1a1a] flex items-center justify-center">
                        <img
                          src="/41598_2019_38966_Fig8_HTML.jpg"
                          alt="Rice plant"
                          className="max-w-full max-h-full w-auto h-auto object-contain"
                          style={{ 
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover'
                          }}
                          onError={(e) => {
                            console.error('Image failed to load')
                            const target = e.target as HTMLImageElement
                            target.style.display = 'none'
                          }}
                          onLoad={() => {
                            console.log('Image loaded successfully')
                          }}
                        />
                      </div>
                    </div>
                  )}
                  <p className="whitespace-pre-wrap break-words font-medium text-sm">
                    {message.content}
                  </p>
                </div>
              </div>
            </div>
          </div>
          )
        })}
      </div>
    </div>
  )
}
