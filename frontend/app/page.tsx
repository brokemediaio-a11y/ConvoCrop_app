'use client'

import { useRouter } from 'next/navigation'
import Link from 'next/link'
import Image from 'next/image'
import GlassSurface from '@/components/GlassSurface'
import SplitText from '@/components/SplitText'
import FloatingLines from '@/components/FloatingLines'
import AnimatedChat from '@/components/AnimatedChat'
import AboutSection from '@/components/AboutSection'

export default function Home() {
  const router = useRouter()

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background Effect */}
      <div className="fixed inset-0 w-full h-full z-0 bg-black">
        <FloatingLines
          enabledWaves={['top']}
          lineCount={5}
          lineDistance={5}
          bendRadius={5}
          bendStrength={-0.5}
          interactive
          parallax
          linesGradient={['#a8ffbf', '#00a71b', '#0b5b2a']}
          mixBlendMode="screen"
        />
      </div>

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col px-4 py-6 md:py-10">
        {/* Header */}
        <header className="w-full flex justify-center">
          <GlassSurface
            width="100%"
            height={64}
            borderRadius={999}
            borderWidth={0.12}
            brightness={32}
            opacity={0.9}
            blur={18}
            displace={3}
            backgroundOpacity={0.1}
            saturation={1.1}
            distortionScale={-140}
            redOffset={0}
            greenOffset={20}
            blueOffset={46}
            xChannel="R"
            yChannel="G"
            mixBlendMode="normal"
            className="max-w-6xl px-4 md:px-8 flex items-center"
          >
            <div className="relative w-full flex items-center">
              {/* Logo - pinned to far left */}
              <div className="flex items-center gap-3 md:gap-4">
                <div className="relative w-32 h-8 md:w-40 md:h-10">
                  <Image
                    src="/logos/convcrop logo.png"
                    alt="Convo Crop logo"
                    fill
                    className="object-contain"
                    priority
                  />
                </div>
              </div>

              {/* Centered navigation */}
              <nav className="absolute inset-x-0 flex items-center justify-center gap-4 md:gap-8 text-sm md:text-base font-medium text-[#dedede]">
                <button
                  type="button"
                  onClick={() => router.push('/')}
                  className="hover:text-[#00a71b] transition-colors"
                >
                  Home
                </button>
                <button
                  type="button"
                  onClick={() => router.push('/about')}
                  className="hidden sm:inline-block hover:text-[#00a71b] transition-colors"
                >
                  About us
                </button>
                <button
                  type="button"
                  onClick={() => router.push('/chat')}
                  className="hover:text-[#00a71b] transition-colors"
                >
                  Chat
                </button>
                <button
                  type="button"
                  onClick={() => router.push('/api')}
                  className="hidden sm:inline-block hover:text-[#00a71b] transition-colors"
                >
                  API
                </button>
                <button
                  type="button"
                  onClick={() => router.push('/docs')}
                  className="hidden sm:inline-block hover:text-[#00a71b] transition-colors"
                >
                  Docs
                </button>
                <button
                  type="button"
                  onClick={() => router.push('/research')}
                  className="hidden sm:inline-block hover:text-[#00a71b] transition-colors"
                >
                  Research
                </button>
              </nav>
            </div>
          </GlassSurface>
        </header>

        {/* Main Content */}
        <div className="flex flex-col flex-1 justify-center max-w-7xl mx-auto w-full">
          <div className="flex flex-col lg:flex-row items-center gap-8 lg:gap-4 w-full">
            {/* Left hero text */}
            <div className="flex-1 max-w-2xl flex flex-col items-start pl-4 md:pl-12 lg:pl-16">
              <SplitText
                text="Convo Crop"
                tag="h1"
                className="text-6xl md:text-8xl font-extrabold text-[#dedede] drop-shadow-light"
                delay={50}
                duration={1.25}
                ease="power3.out"
                splitType="chars"
                from={{ opacity: 0, y: 40 }}
                to={{ opacity: 1, y: 0 }}
                threshold={0.1}
                rootMargin="-100px"
                textAlign="left"
              />

              <p className="mt-6 max-w-xl text-sm md:text-base text-[#dedede] opacity-80">
                an intelligent conversational assistant designed to help farmers detect and diagnose
                crop diseases at an early stage. By simply sharing images, farmers receive instant,
                AI-powered insights about potential issues, treatment recommendations, and preventive
                measures. We replace the need for constant field visits by agronomists, making
                expert-level crop analysis accessible anytime, anywhere empowering farmers to protect
                their yield, reduce losses, and make informed decisions with confidence.
              </p>

              <div className="mt-8 flex flex-wrap gap-4">
                <GlassSurface
                  width={160}
                  height={50}
                  borderRadius={25}
                  onClick={() => router.push('/chat')}
                  className="cursor-pointer hover:opacity-90 transition-opacity"
                >
                  <span className="text-[#dedede] font-medium text-lg">Get Started</span>
                </GlassSurface>

                <GlassSurface
                  width={160}
                  height={50}
                  borderRadius={25}
                  onClick={() => router.push('/contact')}
                  className="cursor-pointer hover:opacity-90 transition-opacity"
                >
                  <span className="text-[#dedede] font-medium text-lg">Contact us</span>
                </GlassSurface>
              </div>
            </div>

            {/* Right hero animated chat */}
            <div className="hidden lg:flex flex-1 flex-col items-center justify-center">
              <div
                className="flex items-center justify-center"
                style={{ minWidth: '400px', minHeight: '500px', height: '560px' }}
              >
                <AnimatedChat />
              </div>
              
              {/* Logos - reduced by 3x and placed under chat */}
              <div className="flex items-center justify-center gap-3 md:gap-4 mt-4">
                <div className="relative w-11 h-11 md:w-14 md:h-14">
                  <Image
                    src="/logos/bahria-university-logo.png"
                    alt="Bahria University"
                    fill
                    className="object-contain"
                    priority
                  />
                </div>

                <div className="text-xs md:text-xl font-light text-white animate-rotate-x">×</div>

                <div className="relative w-11 h-11 md:w-14 md:h-14">
                  <Image
                    src="/logos/image-removebg-preview.png"
                    alt="NCAI"
                    fill
                    className="object-contain"
                    priority
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* About Section */}
      <AboutSection
        teamMembers={[
          {
            name: 'Saad Hassan',
            image: '/2faa39a1-76f0-403f-8896-742ac9c4a5ab.jpg',
            linkedin: 'https://linkedin.com/in/saad-hassan',
          },
          {
            name: 'Qurat ul ain Fatima',
            image: '/a9761351-ac7e-4c32-9a7c-14fddaabfce2.jpg',
            linkedin: 'https://linkedin.com/in/qurat-ul-ain-fatima',
          },
          {
            name: 'Anwar Iqbal Sanjrani',
            image: '/1754662920733.jpg',
            linkedin: 'https://linkedin.com/in/anwar-iqbal-sanjrani',
          },
          {
            name: 'Dr. Asfand e yar',
            image: '/images (2).jpg',
            linkedin: 'https://linkedin.com/in/dr-asfand-e-yar',
          },
        ]}
      />
    </div>
  )
}