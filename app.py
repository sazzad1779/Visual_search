import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from io import BytesIO
from PIL import Image
import uvicorn
from src.image_database import ImageDatabase

# Initialize FastAPI
app = FastAPI()

# Define directories
DATASET_DIR = "archive"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

# Initialize database
image_db = ImageDatabase(dataset_dir=DATASET_DIR, static_dir=STATIC_DIR)
image_db.insert_new_images(batch_size=500)  # Adjust batch size for efficiency
    
# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    """Render the main page for image search."""
    return templates.TemplateResponse("index.html", {"request": request, "query_image": None, "results": []})


@app.post("/search/")
async def search_similar_images(request: Request, file: UploadFile = File(...), top_k: int = 5, threshold: float = 0.65):
    """Handles image upload, performs feature extraction, and retrieves similar images."""
    image_db.clear_static_directory(STATIC_DIR)  # Clear previous results

    image = Image.open(BytesIO(await file.read())).convert('RGB')
    query_image_path = f"/static/query.jpg"
    image.save(os.path.join(STATIC_DIR, "query.jpg"))  # Save query image

    # Perform search
    results = image_db.search_similar_images(image, top_k=top_k, threshold=threshold)

    return templates.TemplateResponse("index.html", {"request": request, "query_image": query_image_path, "results": results})


if __name__ == "__main__":
    # Efficient batch processing before starting the server
    print("Database updated. Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
