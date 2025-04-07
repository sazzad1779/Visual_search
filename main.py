import os
from dotenv import load_dotenv
from src.api_server import app

# Load .env variables
load_dotenv(override=True)

# Set default host/port
PORT = int(os.environ.get("PORT", 4000))
HOST = os.environ.get("HOST", "0.0.0.0")

# This is only used when running via `python main.py`
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Running development server on http://{HOST}:{PORT}")
    uvicorn.run(
        "src.api_server:app",  # as string for reload
        host=HOST,
        port=PORT,
        reload=True
    )

# 👇 IMPORTANT: `app` must be visible at module level
# So Gunicorn can import like: gunicorn -k uvicorn.workers.UvicornWorker main:app
__all__ = ["app"]
