import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Rice Disease Detection',
  description: 'AI-powered rice disease detection chatbot',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
