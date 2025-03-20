import os
import chromadb
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import uvicorn
from io import BytesIO
import shutil
# Define directories
dataset_dir = "archive"  # Dataset images
static_dir = "static"  # Query and copied images

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_full")
collection = chroma_client.get_or_create_collection(name="image_vectors")

# Load Pretrained Model
class FeatureExtractor:
    def __init__(self):
        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final classification layer
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, image: Image.Image) -> np.ndarray:
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(image).squeeze().numpy()
        return features.flatten()

# Initialize Feature Extractor
extractor = FeatureExtractor()

# Function to get all images in database
def get_all_images_in_database():
    existing_records = collection.get()
    return set(record["path"] for record in existing_records["metadatas"]) if "metadatas" in existing_records else set()

# Identify new images and add them to the database
existing_images = get_all_images_in_database()
all_images = set()

for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(root, file)
            all_images.add(img_path)
            
            # Check if image is new
            if img_path not in existing_images:
                image = Image.open(img_path).convert('RGB')
                feature_vector = extractor.extract(image)
                feature_vector = feature_vector / np.linalg.norm(feature_vector)  # Normalize to unit norm
                feature_vector = feature_vector.tolist()
                image_id = f"img_{file}"
                collection.add(ids=[image_id], embeddings=[feature_vector], metadatas=[{"path": img_path}])
                print(f"Added {img_path} to the database.")
            else:
                print(f"Skipping {img_path}, already in database.")
# Function to clear static directory
def clear_static_directory():
    for filename in os.listdir(static_dir):
        file_path = os.path.join(static_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
# # FastAPI Server
app = FastAPI()

app.mount("/static", StaticFiles(directory=static_dir), name="static")  # For query and copied images
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "query_image": None, "results": []})

@app.post("/search/")
async def search_similar_images(request: Request, file: UploadFile = File(...), top_k: int = 5, threshold: float = 0.75):
    clear_static_directory()
    image = Image.open(BytesIO(await file.read())).convert('RGB')
    image_path = f"/static/query.jpg"  # Save query image
    image.save(f"{static_dir}/query.jpg")
    
    # query_vector = extractor.extract(image).tolist()
    query_vector = extractor.extract(image)
    query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize to unit norm
    query_vector = query_vector.tolist()

    results = collection.query(query_embeddings=[query_vector], n_results=top_k*3 )
    
    # Filter results based on similarity threshold
    filtered_results = []
    for i, dist in enumerate(results["distances"][0]):
            similarity = 1 - dist  # Convert distance to similarity
            print(similarity)
            if 0 <= similarity <= 1 and similarity >= threshold:
                original_image_path = results["metadatas"][0][i]["path"]
                copied_image_path = os.path.join(static_dir, os.path.basename(original_image_path))
                shutil.copy(original_image_path, copied_image_path)  # Copy matching image to static
                filtered_results.append(f"/static/{os.path.basename(original_image_path)}")
    
    return templates.TemplateResponse("index.html", {"request": request, "query_image": image_path, "results": filtered_results})

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
