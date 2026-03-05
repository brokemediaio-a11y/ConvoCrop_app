# Testing Guide - Rice Disease Detection API

## Method 1: Using the Swagger UI (Easiest)

1. **Start the server:**
   ```bash
   python run_server.py
   ```

2. **Open the API docs:**
   - Go to: http://localhost:8000/docs
   - You'll see the Swagger UI interface

3. **Test the Health Endpoint first:**
   - Click on `GET /api/health`
   - Click "Try it out"
   - Click "Execute"
   - Verify `model_loaded: true`

4. **Test the Chat Endpoint:**
   - Click on `POST /api/chat`
   - Click "Try it out"
   - You'll see a JSON editor

5. **Convert your image to base64:**
   - Use an online tool: https://www.base64-image.de/
   - Or use Python:
     ```python
     import base64
     with open("your_image.jpg", "rb") as f:
         base64_str = base64.b64encode(f.read()).decode('utf-8')
     print(base64_str)
     ```

6. **Fill in the JSON payload:**
   ```json
   {
     "image": "iVBORw0KGgoAAAANSUhEUgAA... (your base64 string here)",
     "question": "What disease does this rice leaf have?",
     "max_new_tokens": 200
   }
   ```

7. **Click "Execute"** and wait for the response

## Method 2: Using the Test Script

1. **Make sure you have `requests` installed:**
   ```bash
   pip install requests
   ```

2. **Run the test script:**
   ```bash
   # Test health endpoint
   python test_api.py
   
   # Test with an image
   python test_api.py path/to/your/image.jpg
   
   # Test with custom question
   python test_api.py path/to/your/image.jpg "What disease is affecting this rice plant?"
   ```

## Method 3: Using curl (Command Line)

1. **Convert image to base64 first:**
   ```bash
   # Windows PowerShell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("image.jpg"))
   
   # Linux/Mac
   base64 -i image.jpg
   ```

2. **Send the request:**
   ```bash
   curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d "{\"image\": \"YOUR_BASE64_STRING_HERE\", \"question\": \"What disease does this rice leaf have?\"}"
   ```

## Method 4: Using Python Requests Directly

```python
import base64
import requests

# Load and encode image
with open("test_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Prepare payload
payload = {
    "image": image_base64,
    "question": "What disease does this rice leaf have?",
    "max_new_tokens": 200
}

# Send request
response = requests.post(
    "http://localhost:8000/api/chat",
    json=payload,
    timeout=300  # 5 minutes for CPU inference
)

# Print response
print(response.json())
```

## Method 5: Using the Upload Endpoint

If you prefer file upload instead of base64:

1. **In Swagger UI:**
   - Go to `POST /api/upload`
   - Click "Try it out"
   - Click "Choose File" and select your image
   - Enter your question in the "question" field
   - Click "Execute"

2. **Using curl:**
   ```bash
   curl -X POST "http://localhost:8000/api/upload" \
     -F "file=@path/to/your/image.jpg" \
     -F "question=What disease does this rice leaf have?"
   ```

## Example Payload Structure

### Simple Request (No Conversation History)
```json
{
  "image": "base64_encoded_image_string",
  "question": "What disease does this rice leaf have?",
  "max_new_tokens": 200
}
```

### Request with Conversation History
```json
{
  "image": "base64_encoded_image_string",
  "question": "What are the symptoms?",
  "conversation_history": [
    {
      "role": "user",
      "content": "What disease does this rice leaf have?"
    },
    {
      "role": "assistant",
      "content": "This rice leaf shows symptoms of bacterial blight..."
    }
  ],
  "max_new_tokens": 200
}
```

## Expected Response

```json
{
  "response": "The rice leaf shows symptoms of bacterial blight. The disease is characterized by...",
  "disease_detected": "blight"
}
```

## Troubleshooting

### Image Too Large
- Base64 encoding increases file size by ~33%
- For large images, consider resizing before encoding
- Maximum recommended: 5-10MB original image

### Request Timeout
- CPU inference can take 30 seconds - 2 minutes
- Increase timeout in your client
- Check server logs for progress

### Invalid Base64
- Make sure the base64 string doesn't include data URL prefix
- If using `data:image/jpeg;base64,` prefix, remove it before sending

### Model Not Loaded
- Check `/api/health` endpoint
- Verify model checkpoint files exist
- Check server logs for errors

## Quick Image to Base64 Converter (Python)

Save this as `encode_image.py`:

```python
import base64
import sys

if len(sys.argv) < 2:
    print("Usage: python encode_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
with open(image_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')
    print(encoded)
```

Usage:
```bash
python encode_image.py test_image.jpg > base64_output.txt
```
