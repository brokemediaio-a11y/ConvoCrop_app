# Quick Start Guide

## Step 1: Install Dependencies

```bash
cd frontend
npm install
```

## Step 2: Make Sure Backend is Running

The frontend needs the backend API to be running on `http://localhost:8000`.

If it's not running, start it:
```bash
cd ../backend
python run_server.py
```

## Step 3: Start the Frontend

```bash
npm run dev
```

The app will be available at: http://localhost:3000

## Step 4: Use the App

1. **Upload Image**: 
   - Click "Choose Image" or drag & drop a rice leaf image
   - Supported formats: JPEG, PNG, WebP (Max 10MB)

2. **Ask Questions**:
   - Type questions like "What disease does this rice leaf have?"
   - Press Enter or click "Send"
   - Wait for AI response (may take 2-5 seconds on GPU, 30s-2min on CPU)

3. **Continue Conversation**:
   - Ask follow-up questions
   - The AI remembers the conversation context

4. **Start Over**:
   - Click "New Image" to upload a different image
   - Click "Clear Chat" to reset the conversation

## Troubleshooting

### Port Already in Use
If port 3000 is already in use:
```bash
# Use a different port
npm run dev -- -p 3001
```

### Backend Connection Error
- Make sure backend is running on http://localhost:8000
- Check backend health: http://localhost:8000/api/health
- Verify CORS is enabled in backend (it should be by default)

### Image Upload Issues
- Make sure image is less than 10MB
- Supported formats: JPEG, PNG, WebP
- Try a different image if one doesn't work

### Slow Responses
- GPU inference: 2-5 seconds (normal)
- CPU inference: 30 seconds - 2 minutes (normal)
- First request is always slower (warmup)

## Build for Production

```bash
npm run build
npm start
```

This creates an optimized production build.
