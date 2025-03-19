from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import uvicorn
import json
from src.Image_db import ImageDatabase



# Define Pydantic models for request validation
class Image(BaseModel):
    url: str
    image_id: str = Field(alias="_id")  # Use a regular Python name but alias it to "_id" for JSON
    blurDataURL: Optional[str] = None
    
    class Config:
        populate_by_name = True  # Allow accessing the field by both names
        extra = "allow"  # Allow extra fields in the JSON

class ProductUpdate(BaseModel):
    product_id: str = Field(alias="_id")  # This maps _id in the JSON to product_id in Python
    images: List[Image]
    category: Optional[str] = None
    subCategory: Optional[str] = None

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class ImageUpdate(BaseModel):
    images: List[Image]


class SearchResult(BaseModel):
    product_id: str = Field(alias="_id")
    image_id: str
    category_id: Optional[str] = None
    subcategory_id: Optional[str] = None
    similarity: float
    distance: float
    class Config:
        populate_by_name = True  # Allow accessing the field by both names
        allow_population_by_field_name = True



class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float


# Create FastAPI app
app = FastAPI(title="Visual Search API", version="1.0.0")

# Initialize image database
# You can configure the model via environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")

# Initialize the database with ChromaDB
image_db = ImageDatabase(model_name=MODEL_NAME, persist_directory=CHROMA_DIR)
print(f"Initialized ChromaDB with model {MODEL_NAME} at {CHROMA_DIR}")


# ✅ API to update product information
@app.post("/api/update-product")
async def update_product(data: ProductUpdate):
    """Update product information in the visual search database"""
    try:
        print(f"Received update for product: {data.product_id}")
        print(data)
        # Convert ObjectId to string if needed
        product_id = str(data.product_id)
        category_id = str(data.category) if data.category else None
        subcategory_id = str(data.subCategory) if data.subCategory else None
        
        # Update product in the database
        success, num_added, failed = image_db.insert_product(
            product_id=product_id,
            images=data.images,
            category_id=category_id,
            subcategory_id=subcategory_id
        )
        
        if not success:
            return {
                "message": "Product update processed but no images were added",
                "product_id": product_id,
                "images_added": 0,
                "failed_images": failed
            }
            
        return {
            "message": "Product update received",
            "product_id": product_id,
            "images_added": num_added,
            "failed_images": failed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update product: {str(e)}")


# ✅ API to delete a product from vector DB
@app.delete("/api/delete-product/{product_id}")
async def delete_product(product_id: str):
    """Delete a product from the visual search database"""
    try:
        print(f"Deleting product from vector DB: {product_id}")
        
        # Delete from database
        success = image_db.delete_product(product_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return {"message": "Product deleted from visual search", "product_id": product_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete product: {str(e)}")


# ✅ API to update images in vector DB
@app.post("/api/update-images/{product_id}")
async def update_images(product_id: str, data: ImageUpdate):
    """Update images for a specific product in the visual search database"""
    try:
        print(f"Updating images for product: {product_id}")
        
        # Update images in database
        success, num_added, failed = image_db.update_product(
            product_id=product_id,
            images=data.images
        )
        
        if not success and len(failed) == len(data.images):
            raise HTTPException(status_code=500, detail="Failed to process any images")
        
        return {
            "message": "Images updated in visual search",
            "product_id": product_id,
            "images_added": num_added,
            "failed_images": failed
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update images: {str(e)}")


# ✅ API to search by image URL
@app.get("/api/search")
async def search_by_image(image_url: str, limit: int = 10):
    """Search for similar products by image URL"""
    import time
    
    try:
        start_time = time.time()
        
        # Query the database
        results = image_db.query_image(image_url, k=limit)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        if not results:
            return {"results": [], "query_time_ms": query_time_ms}
        
        return {"results": results, "query_time_ms": query_time_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# API to get product information
@app.get("/api/product/{product_id}")
async def get_product(product_id: str):
    """Get product information from the database"""
    try:
        if product_id not in image_db.product_map:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Get images for this product
        images = image_db.get_product_images(product_id)
        
        # Get category and subcategory from any image if available
        category_id = None
        subcategory_id = None
        
        if images:
            # Query the first image to get more details
            image_id = image_db.product_map[product_id][0]
            result = image_db.collection.get(
                ids=[image_id],
                include=["metadatas"]
            )
            
            if result and 'metadatas' in result and result['metadatas']:
                metadata = result['metadatas'][0]
                category_id = metadata.get('category_id', '')
                subcategory_id = metadata.get('subcategory_id', '')
        
        return {
            "product_id": product_id,
            "category_id": category_id,
            "subcategory_id": subcategory_id,
            "images": images,
            "image_count": len(images)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get product: {str(e)}")


# API to get database stats
@app.get("/api/stats")
async def get_stats():
    """Get statistics about the vector database"""
    try:
        stats = image_db.get_stats()
        stats["chroma_dir"] = CHROMA_DIR
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")



if __name__ == "__main__":
    # Run the server
    PORT = int(os.environ.get("PORT", 4000))
    HOST = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting visual search API server on {HOST}:{PORT}")
    print(f"Using model: {MODEL_NAME}")
    
    uvicorn.run(app, host=HOST, port=PORT)