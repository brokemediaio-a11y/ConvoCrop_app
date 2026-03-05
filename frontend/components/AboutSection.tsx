'use client'

import MagicBento from './MagicBento'
import TeamCarousel from './TeamCarousel'

interface TeamMember {
  name: string
  image: string
  linkedin: string
}

interface AboutSectionProps {
  teamMembers?: TeamMember[]
}

export default function AboutSection({ teamMembers = [] }: AboutSectionProps) {
  const aboutCards = [
    {
      color: '#1a1a1a',
      title: 'Our Mission',
      description:
        'To empower farmers worldwide with accessible AI-powered crop disease detection. We aim to reduce agricultural losses, increase food security, and make expert-level crop analysis available to every farmer, regardless of location or resources.',
      label: 'Mission',
    },
    {
      color: '#1a1a1a',
      title: 'Our Vision',
      description:
        'A future where every farmer has instant access to accurate disease diagnosis and treatment recommendations. We envision a world where technology bridges the gap between agricultural expertise and field-level implementation, ensuring sustainable farming practices globally.',
      label: 'Vision',
    },
    {
      color: '#1a1a1a',
      title: 'Our Achievements',
      description:
        'Successfully deployed FastVLM 1.5B model for rice disease detection with 95%+ accuracy. Developed real-time chat interface enabling instant disease diagnosis. Partnered with Bahria University and NCAI to bring AI-powered agricultural solutions to farmers across Pakistan.',
      label: 'Achievements',
    },
  ]

  return (
    <section className="w-full py-16 md:py-24 px-4 md:px-8 relative">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-4xl md:text-6xl font-extrabold text-[#dedede] text-center mb-12 md:mb-16">
          About Us
        </h2>

        {/* Mission, Vision, Achievements Cards */}
        <div className="mb-16 md:mb-24">
          <MagicBento
            textAutoHide={false}
            enableStars={true}
            enableSpotlight={true}
            enableBorderGlow={true}
            enableTilt={true}
            enableMagnetism={true}
            clickEffect={true}
            spotlightRadius={250}
            particleCount={12}
            glowColor="0, 167, 27"
            disableAnimations={false}
            cards={aboutCards}
          />
        </div>

        {/* Team Section */}
        {teamMembers.length > 0 && (
          <div className="mt-16 md:mt-24">
            <h3 className="text-3xl md:text-4xl font-extrabold text-[#dedede] text-center mb-12">
              Our Team
            </h3>
            <TeamCarousel members={teamMembers} autoPlay={true} autoPlayInterval={4000} />
          </div>
        )}
      </div>
    </section>
  )
}
