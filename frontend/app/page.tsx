'use client'

import { useRouter } from 'next/navigation'
import Link from 'next/link'
import Image from 'next/image'
import GlassSurface from '@/components/GlassSurface'
import SplitText from '@/components/SplitText'
import FloatingLines from '@/components/FloatingLines'
import PlantViewer from '@/components/PlantViewer'
import AboutSection from '@/components/AboutSection'
import { CONVOCROP_LOGO_SRC } from '@/lib/assets'

export default function Home() {
  const router = useRouter()

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background Effect */}
      <div className="fixed inset-0 w-full h-full z-0 bg-[#ebebeb]">
        <FloatingLines
          enabledWaves={['top']}
          lineCount={5}
          lineDistance={5}
          bendRadius={5}
          bendStrength={-0.5}
          interactive
          parallax
          linesGradient={['#a8ffbf', '#00a71b', '#0b5b2a']}
          mixBlendMode="multiply"
        />
      </div>

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col px-4 py-6 md:py-10">
        {/* 3D Plant - covers entire hero section, behind text */}
        <div className="hidden lg:block absolute inset-0 z-0 pointer-events-none overflow-hidden">
          <PlantViewer />
        </div>

        {/* Header */}
        <header className="w-full flex justify-center relative z-10">
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
              <div className="-ml-2 md:-ml-4 lg:-ml-6 flex items-center gap-2 md:gap-3">
                <div className="relative w-32 h-8 md:w-40 md:h-10">
                  <Image
                    src={CONVOCROP_LOGO_SRC}
                    alt="Convo Crop logo"
                    fill
                    className="object-contain"
                    priority
                  />
                </div>

                {/* Separator + Partner logos */}
                <div className="hidden md:flex items-center gap-2">
                  <div className="h-8 w-px bg-[#252525]/30" />

                  <div className="relative w-8 h-8 md:w-9 md:h-9">
                    <Image
                      src="/logos/bahria-university-logo.png"
                      alt="Bahria University"
                      fill
                      className="object-contain"
                      priority
                    />
                  </div>

                  <div className="text-sm md:text-base font-light text-[#252525]">+</div>

                  <div className="relative w-9 h-9 md:w-10 md:h-10">
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

              {/* Centered navigation */}
              <nav className="absolute left-1/2 -translate-x-1/2 flex items-center justify-center gap-4 md:gap-8 text-sm md:text-base font-medium text-[#252525]">
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
        <div className="flex flex-col flex-1 justify-center max-w-7xl mx-auto w-full relative z-10">
          <div className="flex flex-col lg:flex-row items-center gap-8 lg:gap-4 w-full">
            {/* Left hero text */}
            <div className="flex-1 max-w-2xl flex flex-col items-start pl-4 md:pl-12 lg:pl-16">
              <SplitText
                text="Convo-Crop"
                tag="h1"
                className="text-6xl md:text-8xl font-extrabold text-[#252525] drop-shadow-light"
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

              <p className="mt-6 max-w-xl text-sm md:text-base text-[#252525] opacity-80 text-justify">
                An AI-powered assistant that helps farmers detect crop diseases early using images.
                Get instant diagnosis, treatment advice, and preventive measures anytime. Empowering
                smarter decisions, higher yields, and reduced losses.
              </p>

              <div className="mt-8 flex flex-wrap gap-4">
                <GlassSurface
                  width={160}
                  height={50}
                  borderRadius={25}
                  onClick={() => router.push('/chat')}
                  className="cursor-pointer hover:opacity-90 transition-opacity"
                >
                  <span className="text-[#252525] font-medium text-lg">Get Started</span>
                </GlassSurface>

                <GlassSurface
                  width={160}
                  height={50}
                  borderRadius={25}
                  onClick={() => router.push('/contact')}
                  className="cursor-pointer hover:opacity-90 transition-opacity"
                >
                  <span className="text-[#252525] font-medium text-lg">Contact us</span>
                </GlassSurface>
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