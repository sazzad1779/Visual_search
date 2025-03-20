import os
from dotenv import load_dotenv
import uvicorn
from src.api_server import app

# Load environment variables from .env file if it exists
load_dotenv()

if __name__ == "__main__":
    # Run the server with reload enabled
    PORT = int(os.environ.get("PORT", 4000))
    HOST = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting visual search API server on {HOST}:{PORT}")
    print(f"Reload is enabled - changes to code will automatically restart the server")
    
    # Enable reload for development
    uvicorn.run(
        "src.api_server:app",  # Use string reference for reload to work
        host=HOST, 
        port=PORT,
        reload=True           # This enables auto-reload on code changes
    )