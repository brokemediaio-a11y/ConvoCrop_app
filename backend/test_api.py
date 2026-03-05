"""Test script for the Rice Disease Detection API."""
import base64
import json
import requests
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded


def test_chat_endpoint(image_path: str, question: str = "What disease does this rice leaf have?", api_url: str = "http://localhost:8000"):
    """Test the /api/chat endpoint."""
    
    # Convert image to base64
    print(f"Loading image: {image_path}")
    try:
        image_base64 = image_to_base64(image_path)
        print(f"Image encoded successfully ({len(image_base64)} characters)")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Prepare request payload
    payload = {
        "image": image_base64,
        "question": question,
        "max_new_tokens": 200
    }
    
    # Send request
    url = f"{api_url}/api/chat"
    print(f"\nSending request to {url}...")
    print(f"Question: {question}")
    
    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout for CPU
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("RESPONSE:")
            print("="*60)
            print(f"Response: {result.get('response', 'N/A')}")
            print(f"Disease Detected: {result.get('disease_detected', 'N/A')}")
            print("="*60)
        else:
            print(f"Error: {response.text}")
    
    except requests.exceptions.Timeout:
        print("Request timed out. The model might be processing on CPU (this can take 30s-2min).")
    except Exception as e:
        print(f"Error sending request: {e}")


def test_health_endpoint(api_url: str = "http://localhost:8000"):
    """Test the /api/health endpoint."""
    url = f"{api_url}/api/health"
    print(f"Testing health endpoint: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Rice Disease Detection API - Test Script")
    print("="*60)
    
    # Test health first
    print("\n1. Testing health endpoint...")
    test_health_endpoint()
    
    # Test chat endpoint if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        question = sys.argv[2] if len(sys.argv) > 2 else "What disease does this rice leaf have?"
        
        if not Path(image_path).exists():
            print(f"\nError: Image file not found: {image_path}")
            sys.exit(1)
        
        print(f"\n2. Testing chat endpoint with image: {image_path}")
        test_chat_endpoint(image_path, question)
    else:
        print("\n2. To test chat endpoint, provide an image path:")
        print("   python test_api.py <path_to_image> [question]")
        print("\n   Example:")
        print("   python test_api.py test_image.jpg")
        print("   python test_api.py test_image.jpg 'What disease is this?'")
