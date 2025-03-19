import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    """Main entry point for the visual search application"""
    parser = argparse.ArgumentParser(description="E-Commerce Visual Search Engine")
    
    # Add command line arguments
    parser.add_argument(
        "--model", 
        type=str, 
        default=os.environ.get("MODEL_NAME", "efficientnet"),
        choices=["resnet50", "efficientnet", "mobilenet"],
        help="Feature extraction model to use"
    )
    
    parser.add_argument(
        "--chroma-dir", 
        type=str, 
        default=os.environ.get("CHROMA_DIR", "./chroma_db"),
        help="Directory for ChromaDB persistence"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.environ.get("PORT", 4000)),
        help="Port to run the server on"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to run the server on"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_NAME"] = args.model
    os.environ["CHROMA_DIR"] = args.chroma_dir
    os.environ["PORT"] = str(args.port)
    os.environ["HOST"] = args.host
    
    # Import and run the API server
    # We import here to ensure environment variables are set first
    import uvicorn
    from src.api_server import app
    
    print(f"Starting visual search API server on {args.host}:{args.port}")
    print(f"Using model: {args.model}")
    print(f"ChromaDB directory: {args.chroma_dir}")
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()