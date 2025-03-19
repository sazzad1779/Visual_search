# Visual Search API

A powerful image similarity search API that uses vector embeddings and neural networks to find visually similar products.

## Overview

This project implements a visual search system that allows users to search for products using images. The system extracts features from product images using deep learning models and stores them in a vector database (ChromaDB). Users can then query the system with new images to find visually similar products.

## Features

- **Multiple Search Methods**: 
  - Search by image URL
  - Search by uploading an image file
  - Search by base64-encoded image data
  - Search by using the device camera
  
- **Product Management**:
  - Add/update products and images
  - Delete products
  - Batch insert multiple products
  - Update images for existing products
  
- **Supported Feature Extraction Models**:
  - EfficientNet (default)
  - ResNet50
  - MobileNetV3
  
- **Vector Database**:
  - Uses ChromaDB for efficient similarity search
  - Persistent storage of vector embeddings
  - Fast querying with customizable result limit

## Installation

### Prerequisites

- Python 3.8+
- FastAPI
- PyTorch
- ChromaDB

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sazzad1779/Visual_search.git
   cd Visual_search
   git checkout version_1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (optional):
   ```bash
   ## create an .env variable
   MODEL_NAME=efficientnet  # Options: efficientnet, resnet50, mobilenet
   CHROMA_DIR=./chroma_db   # Directory for storing vector database
   PORT=4000                # API port
   HOST=0.0.0.0             # API host
   ```

## Usage

### Starting the Server

Run the server with:

```bash
python main.py
```

### API Endpoints

#### Image Search

- **GET** `/api/search?image_url={url}&limit={n}`: Search by image URL
- **POST** `/api/search/base64`: Search by base64-encoded image
- **POST** `/api/search/upload`: Search by uploaded image file

#### Product Management

- **POST** `/api/update-product`: Add or update a product
- **DELETE** `/api/delete-product/{product_id}`: Delete a product
- **POST** `/api/batch-insert-products`: Add multiple products at once
- **DELETE** `/api/delete-all`: Delete all products from the database

#### Information

- **GET** `/api/product/{product_id}`: Get product information
- **GET** `/api/stats`: Get database statistics

### Request/Response Examples

#### Search by Image URL

Request:
```
GET /api/search?image_url=https://example.com/image.jpg&limit=10
```

Response:
```json
{
  "results": [
    {
      "product_id": "123",
      "image_id": "img456",
      "category_id": "clothing",
      "subcategory_id": "shirts",
      "similarity": 0.95,
      "distance": 0.05
    },
    ...
  ],
  "query_time_ms": 125.4
}
```

#### Update Product

Request:
```json
POST /api/update-product
{
  "_id": "product123",
  "category": "clothing",
  "subCategory": "shirts",
  "images": [
    {
      "_id": "image1",
      "url": "https://example.com/image1.jpg"
    }
  ]
}
```

Response:
```json
{
  "message": "Product update received",
  "product_id": "product123",
  "images_added": 1,
  "failed_images": []
}
```

## Testing

A test UI is included to easily test the visual search functionality:

1. Open `ui_test.html` in a web browser
2. Use the file upload, camera, or URL input to test the search functionality
3. View search results including similarity scores

## Exposing the API with ngrok (optional)

To make your API accessible from the internet:

1. Install ngrok: https://ngrok.com/download
2. Authenticate with ngrok:
   ```bash
   ngrok authtoken YOUR_AUTH_TOKEN
   ```
3. Expose your API:
   ```bash
   ngrok http 4000
   ```
4. Use the provided ngrok URL to access your API publicly

## Architecture

The system consists of several key components:

- **FastAPI Server**: Handles HTTP requests and API endpoints
- **Feature Extractor**: Neural network models that convert images to feature vectors
- **Image Database**: ChromaDB vector database for storing and querying image embeddings
- **Product Manager**: Functions for managing product data and image associations

## Project Structure

```
visual-search-api/
├── api_server.py          # FastAPI server implementation
├── chroma_db/             # Vector database storage
├── src/
│   ├── feature_extractor.py  # Image feature extraction
│   ├── image_db.py        # Vector database wrapper
│   └── connect_mongodb.py # MongoDB connector (optional)
├── test/                # Static files for UI
│   └── ui_test.html  # Test UI
└── requirements.txt       # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
