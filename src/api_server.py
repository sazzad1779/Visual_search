
from fastapi import FastAPI, HTTPException, UploadFile, File, Form,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import uvicorn
from src.Image_db import ImageDatabase
import base64
import io
# from PIL import Image
import time


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
import asyncio

# Define Pydantic models for batch insert
class BatchProductInsert(BaseModel):
    products: List[ProductUpdate]
    
    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "products": [
                    {
                        "_id": "product123",
                        "category": "clothing",
                        "subCategory": "shirts",
                        "images": [
                            {
                                "_id": "image1",
                                "url": "https://example.com/image1.jpg",
                                "blurDataURL": "https://example.com/image1.jpg"
                            }
                        ]
                    },
                    {
                        "_id": "product456",
                        "category": "accessories",
                        "subCategory": "watches",
                        "images": [
                            {
                                "_id": "image2",
                                "url": "https://example.com/image2.jpg",
                                "blurDataURL": "https://example.com/image1.jpg"
                            }
                        ]
                    }
                ]
            }
        }

# Response model for batch operations
class BatchInsertResponse(BaseModel):
    total_products: int
    successful_products: int
    failed_products: List[Dict[str, Any]]
    total_images_added: int

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
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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


# 1. Search with Base64 image data (fastest for real-time)
@app.post("/api/search/base64")
async def search_by_base64_image(base64_image: str = Form(...), limit: int = Form(10)):
    """Search for similar products using base64 encoded image data"""
    try:
        start_time = time.time()
        
        # Decode base64 image
        try:
            # Remove the data URL prefix if present
            if "base64," in base64_image:
                base64_image = base64_image.split("base64,")[1]
                
            image_data = base64.b64decode(base64_image)
            
            # Query using the image bytes
            results = image_db.query_image_from_bytes(image_data, k=limit)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        query_time_ms = (time.time() - start_time) * 1000
        
        if not results:
            return {"results": [], "query_time_ms": query_time_ms}
        
        return {"results": results, "query_time_ms": query_time_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# 2. Search with direct file upload
@app.post("/api/search/upload")
async def search_by_uploaded_image(file: UploadFile = File(...), limit: int = Form(10)):
    """Search for similar products using uploaded image file"""
    try:
        start_time = time.time()
        
        # Read file content
        image_data = await file.read()
        
        # Query using the bytes directly
        results = image_db.query_image_from_bytes(image_data, k=limit)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        if not results:
            return {"results": [], "query_time_ms": query_time_ms}
        
        return {"results": results, "query_time_ms": query_time_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
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

async def process_product(image_db, product):
    """Process a single product for batch insertion"""
    try:
        product_id = str(product.product_id)
        category_id = str(product.category) if product.category else None
        subcategory_id = str(product.subCategory) if product.subCategory else None
        
        success, num_added, failed = image_db.insert_product(
            product_id=product_id,
            images=product.images,
            category_id=category_id,
            subcategory_id=subcategory_id
        )
        
        return {
            "product_id": product_id,
            "success": success,
            "images_added": num_added,
            "failed_images": failed
        }
    except Exception as e:
        return {
            "product_id": str(product.product_id),
            "success": False,
            "error": str(e),
            "images_added": 0
        }

@app.post("/api/batch-insert-products", response_model=BatchInsertResponse)
async def batch_insert_products(data: BatchProductInsert, background_tasks: BackgroundTasks):
    """Insert multiple products into the visual search database in a single API call"""
    try:
        products = data.products
        
        if not products:
            return {
                "total_products": 0,
                "successful_products": 0,
                "failed_products": [],
                "total_images_added": 0
            }
        
        print(f"Batch inserting {len(products)} products")
        
        # Process each product concurrently
        tasks = [process_product(image_db, product) for product in products]
        results = await asyncio.gather(*tasks)
        
        # Collect statistics
        successful_products = [r for r in results if r["success"]]
        failed_products = [r for r in results if not r["success"]]
        total_images_added = sum(r["images_added"] for r in results)
        
        return {
            "total_products": len(products),
            "successful_products": len(successful_products),
            "failed_products": failed_products,
            "total_images_added": total_images_added
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch insert failed: {str(e)}")
# Add this endpoint to your api_server.py file

@app.delete("/api/delete-all")
async def delete_all_vectors():
    """Delete all records from the vector database"""
    try:
        print("Deleting all records from vector database...")
        
        # Get all IDs in the collection
        all_items = image_db.collection.get()
        ids = all_items.get('ids', [])
        
        if not ids:
            return {"message": "Database is already empty", "deleted_count": 0}
        
        # Delete all records from ChromaDB
        image_db.collection.delete(ids=ids)
        
        # Clear the product map
        image_db.product_map = {}
        
        return {
            "message": "All records deleted from vector database",
            "deleted_count": len(ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete all records: {str(e)}")
    
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