"""Server startup script."""
import uvicorn
from app.config import HOST, PORT
from app.main import app

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Rice Disease Detection API Server")
    print("=" * 60)
    print(f"Server will run on http://{HOST}:{PORT}")
    print(f"API docs available at http://{HOST}:{PORT}/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False  # Disable reload for production
    )
