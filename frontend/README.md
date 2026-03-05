# Rice Disease Detection - Frontend

Next.js frontend application for the Rice Disease Detection chatbot.

## Features

- Modern, responsive chat interface
- Image upload with drag & drop support
- Real-time AI responses
- Conversation history
- Disease detection highlights
- Mobile-friendly design

## Getting Started

### Prerequisites

- Node.js 18+ installed
- Backend API running on http://localhost:8000

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create a `.env.local` file (optional):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

If you don't create this file, it will default to `http://localhost:8000`.

### Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

## Usage

1. **Upload Image**: Click or drag & drop a rice leaf image
2. **Ask Questions**: Type questions about the disease
3. **View Responses**: Get AI-powered disease detection and information
4. **Continue Conversation**: Ask follow-up questions for more details

## Project Structure

```
frontend/
├── app/              # Next.js app directory
│   ├── layout.tsx    # Root layout
│   ├── page.tsx      # Home page
│   └── globals.css   # Global styles
├── components/       # React components
│   ├── ChatInterface.tsx
│   ├── ImageUpload.tsx
│   ├── MessageList.tsx
│   └── ChatInput.tsx
├── lib/              # Utilities
│   └── api.ts        # API client
└── public/           # Static assets
```

## Technologies

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## API Integration

The frontend connects to the backend API at:
- `GET /api/health` - Health check
- `POST /api/chat` - Send chat message with image
- `POST /api/upload` - Alternative file upload endpoint

## Responsive Design

The interface is fully responsive and works on:
- Desktop (1920px+)
- Tablet (768px - 1919px)
- Mobile (320px - 767px)
