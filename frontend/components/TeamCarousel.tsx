'use client'

import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import Image from 'next/image'

interface TeamMember {
  name: string
  image: string
  linkedin: string
}

interface TeamCarouselProps {
  members: TeamMember[]
  autoPlay?: boolean
  autoPlayInterval?: number
}

export default function TeamCarousel({
  members,
}: TeamCarouselProps) {
  if (members.length === 0) return null

  return (
    <div className="w-full py-8">
      <div className="flex items-center justify-center gap-8 flex-wrap">
        {members.map((member, index) => (
          <div
            key={index}
            className="flex flex-col items-center gap-4"
          >
            <div className="relative w-32 h-32 md:w-40 md:h-40 rounded-full overflow-hidden border-4 border-[#00a71b]/50 hover:border-[#00a71b] transition-colors">
              <Image
                src={member.image}
                alt={member.name}
                fill
                className="object-cover"
              />
            </div>
            <div className="text-center">
              <h3 className="text-lg md:text-xl font-bold text-[#252525] mb-2">
                {member.name}
              </h3>
              <a
                href={member.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-[#f0f0f0] border border-[#00a71b]/30 hover:border-[#00a71b] hover:bg-[#00a71b]/10 transition-colors"
              >
                <svg
                  className="w-5 h-5 text-[#00a71b]"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
