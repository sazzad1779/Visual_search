import chromadb
from typing import List, Dict, Any, Optional, Tuple
from src.feature_extractor import get_feature_extractor


class ImageDatabase:
    """Vector database for storing and querying image features using ChromaDB"""
    
    def __init__(self, model_name="efficientnet", persist_directory="./chroma_db"):
        """Initialize the image database with the specified model"""
        self.feature_extractor = get_feature_extractor(model_name)
        self.model_name = model_name
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="product_images",
            metadata={"model": model_name}
        )
        
        # Maps for easier lookups
        self.product_map = {}  # Maps product_id to list of image_ids
        self._initialize_maps()
    
    def _initialize_maps(self):
        """Initialize product map from existing database entries"""
        self.product_map = {}
        
        # Query for all items in the collection
        all_items = self.collection.get()
        
        # Build the product map
        for i, item_id in enumerate(all_items.get('ids', [])):
            metadata = all_items.get('metadatas', [])[i]
            product_id = metadata.get('product_id')
            image_id = metadata.get('image_id')
            
            if product_id:
                if product_id not in self.product_map:
                    self.product_map[product_id] = []
                self.product_map[product_id].append(image_id)
    
    def insert_product(self, product_id: str, images: List[Any], 
                  category_id: str = None, subcategory_id: str = None):
        """
        Insert new product images into the vector database with efficient ID matching
        
        Args:
            product_id: Unique identifier for the product
            images: List of image objects with url and image_id attributes
            category_id: Category identifier (optional)
            subcategory_id: Subcategory identifier (optional)
        
        Returns:
            Tuple of (success: bool, num_added: int, failed_images: List[str])
        """
        # Track metrics
        num_added = 0
        failed_images = []
        skipped_images = []
        
        # Extract all image IDs from the incoming batch for efficient lookup
        batch_image_ids = []
        image_objects = {}  # Map to quickly look up image objects by ID
        
        for image in images:
            # Extract image ID using appropriate attribute
            try:
                if hasattr(image, 'image_id'):
                    image_id = image.image_id
                else:
                    continue
                    
                batch_image_ids.append(image_id)
                image_objects[image_id] = image
            except Exception as e:
                print(f"Error extracting image ID: {e}")
        
        # If no valid images in batch, return early
        if not batch_image_ids:
            return (False, 0, [])
        
        # Query ChromaDB only for the specific IDs in this batch
        # This is more efficient than getting all IDs when database is large
        try:
            existing_results = self.collection.get(
                ids=batch_image_ids,
                include=["metadatas"]
            )
            existing_ids = set(existing_results.get('ids', []))
            print(f"Found {len(existing_ids)} already existing images out of {len(batch_image_ids)} in batch")
        except Exception as e:
            print(f"Error checking existing images: {e}")
            existing_ids = set()
        
        # Process each image ID
        for image_id in batch_image_ids:
            # Skip if already in database
            if image_id in existing_ids:
                print(f"Image {image_id} already exists, skipping...")
                skipped_images.append(image_id)
                
                # Make sure the product map is updated
                if product_id not in self.product_map:
                    self.product_map[product_id] = []
                if image_id not in self.product_map[product_id]:
                    self.product_map[product_id].append(image_id)
                
                continue
            
            # Get the corresponding image object
            image = image_objects[image_id]
            
            try:
                # Get image URL
                if hasattr(image, 'url'):
                    image_url = image.url
                else:
                    print(f"Image {image_id} has no URL, skipping...")
                    failed_images.append(image_id)
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_from_url(image_url)
                if features is None:
                    print(f"Feature extraction failed for image {image_id}")
                    failed_images.append(image_id)
                    continue
                
                # Prepare metadata
                metadata = {
                    'product_id': product_id,
                    'image_id': image_id,
                    'category_id': category_id if category_id else '',
                    'subcategory_id': subcategory_id if subcategory_id else '',
                    'image_url': image_url
                }
                
                # Add to ChromaDB
                self.collection.upsert(
                    ids=[image_id],
                    embeddings=[features.tolist()],
                    metadatas=[metadata]
                )
                
                # Update mappings
                if product_id not in self.product_map:
                    self.product_map[product_id] = []
                if image_id not in self.product_map[product_id]:
                    self.product_map[product_id].append(image_id)
                
                num_added += 1
                
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
                failed_images.append(image_id)
        
        print(f"Product {product_id}: Added {num_added} images, skipped {len(skipped_images)}, failed {len(failed_images)}")
        return (num_added > 0, num_added, failed_images)


    # def update_product(self, product_id: str, images: List[Dict[str, str]], 
    #                   category_id: str = None, subcategory_id: str = None):
    #     """
    #     Update product images in the vector database
        
    #     This handles adding new images without duplicating existing ones
        
    #     Args:
    #         product_id: Unique identifier for the product
    #         images: List of image dictionaries with 'url' and '_id' keys
    #         category_id: Category identifier (optional)
    #         subcategory_id: Subcategory identifier (optional)
        
    #     Returns:
    #         Tuple of (success: bool, num_added: int, failed_images: List[str])
    #     """
    #     # Track metrics
    #     num_added = 0
    #     failed_images = []
        
    #     # Get existing image IDs for this product
    #     existing_image_ids = set(self.product_map.get(product_id, []))
        
    #     # Process each image
    #     for image in images:
    #         # Handle both dict and Pydantic model cases
    #         if hasattr(image, 'url') and hasattr(image, 'image_id'):
    #             # It's a Pydantic model
    #             image_url = image.url
    #             image_id = image.image_id
    #         elif isinstance(image, dict):
    #             # It's a dictionary
    #             image_url = image.get('url')
    #             image_id = image.get('image_id')
    #         else:
    #             print(f"Unrecognized image object type: {type(image)}")
    #             continue
            
    #         # Skip if image already exists
    #         if image_id in existing_image_ids:
    #             continue
            
    #         # Extract features
    #         features = self.feature_extractor.extract_from_url(image_url)
    #         if features is None:
    #             failed_images.append(image_id)
    #             continue
            
    #         # Prepare metadata
    #         metadata = {
    #             'product_id': product_id,
    #             'image_id': image_id,
    #             'category_id': category_id if category_id else '',
    #             'subcategory_id': subcategory_id if subcategory_id else '',
    #             'image_url': image_url
    #         }
    #         # Add to ChromaDB
    #         try:
    #             self.collection.upsert(
    #                 ids=[image_id],
    #                 embeddings=[features.tolist()],
    #                 metadatas=[metadata]
    #             )
                
    #             # Update mappings
    #             if product_id not in self.product_map:
    #                 self.product_map[product_id] = []
    #             self.product_map[product_id].append(image_id)
                
    #             num_added += 1
    #         except Exception as e:
    #             print(f"Error adding image {image_id} to ChromaDB: {e}")
    #             failed_images.append(image_id)
        
    #     return (num_added > 0, num_added, failed_images)
    
    def update_product(self, product_id: str, images: List[Any], 
                  category_id: str = None, subcategory_id: str = None):
        """
        Update product images in the vector database
        
        This handles both adding new images and removing deleted ones
        
        Args:
            product_id: Unique identifier for the product
            images: List of image objects with url and image_id attributes
            category_id: Category identifier (optional)
            subcategory_id: Subcategory identifier (optional)
        
        Returns:
            Tuple of (success: bool, num_added: int, failed_images: List[str])
        """
        # Track metrics
        num_added = 0
        failed_images = []
        
        # Get existing image IDs for this product
        existing_image_ids = set(self.product_map.get(product_id, []))
        
        # Extract all incoming image IDs from the update
        incoming_image_ids = set()
        image_objects = {}  # Map to quickly look up image objects by ID
        
        for image in images:
            # Handle both dict and Pydantic model cases
            if hasattr(image, 'url') and hasattr(image, 'image_id'):
                # It's a Pydantic model
                image_url = image.url
                image_id = image.image_id
            elif isinstance(image, dict):
                # It's a dictionary
                image_url = image.get('url')
                image_id = image.get('image_id')
            else:
                print(f"Unrecognized image object type: {type(image)}")
                continue
            
            incoming_image_ids.add(image_id)
            image_objects[image_id] = image
        
        # Find images that were removed (in existing but not in incoming)
        removed_image_ids = existing_image_ids - incoming_image_ids
        
        # Delete removed images from ChromaDB if any
        if removed_image_ids:
            try:
                print(f"Deleting {len(removed_image_ids)} removed images for product {product_id}")
                self.collection.delete(ids=list(removed_image_ids))
                
                # Update the product map
                if product_id in self.product_map:
                    self.product_map[product_id] = [img_id for img_id in self.product_map[product_id] 
                                                if img_id not in removed_image_ids]
            except Exception as e:
                print(f"Error removing deleted images for product {product_id}: {e}")
        
        # Process each new image (add only images that don't exist yet)
        for image_id in incoming_image_ids:
            # Skip if image already exists
            if image_id in existing_image_ids:
                continue
            
            # Get the corresponding image object
            image = image_objects[image_id]
            
            # Get image URL based on object type
            if hasattr(image, 'url'):
                image_url = image.url
            elif isinstance(image, dict):
                image_url = image.get('url')
            else:
                failed_images.append(image_id)
                continue
            
            # Extract features
            features = self.feature_extractor.extract_from_url(image_url)
            if features is None:
                failed_images.append(image_id)
                continue
            
            # Prepare metadata
            metadata = {
                'product_id': product_id,
                'image_id': image_id,
                'category_id': category_id if category_id else '',
                'subcategory_id': subcategory_id if subcategory_id else '',
                'image_url': image_url
            }
            
            # Add to ChromaDB
            try:
                self.collection.upsert(
                    ids=[image_id],
                    embeddings=[features.tolist()],
                    metadatas=[metadata]
                )
                
                # Update mappings
                if product_id not in self.product_map:
                    self.product_map[product_id] = []
                self.product_map[product_id].append(image_id)
                
                num_added += 1
            except Exception as e:
                print(f"Error adding image {image_id} to ChromaDB: {e}")
                failed_images.append(image_id)
        
        success = num_added > 0 or len(removed_image_ids) > 0
        return (success, num_added, failed_images)
    def delete_product(self, product_id: str):
        """
        Delete a product and all its images from the database
        
        Args:
            product_id: Unique identifier for the product
        
        Returns:
            bool: True if product was deleted, False if not found
        """
        if product_id not in self.product_map:
            return False
        
        # Get all image IDs for this product
        image_ids = self.product_map[product_id]
        
        # Delete from ChromaDB
        try:
            self.collection.delete(ids=image_ids)
            del self.product_map[product_id]
            return True
        except Exception as e:
            print(f"Error deleting product {product_id} from ChromaDB: {e}")
            return False
    
    def query_image(self, image_url: str, k: int = 10):
        """
        Query the database with an image URL to find similar products
        
        Args:
            image_url: URL of the query image
            k: Number of results to return
        
        Returns:
            List of similar products with metadata and similarity scores
        """
        # Extract features from the query image
        features = self.feature_extractor.extract_from_url(image_url)
        if features is None:
            return []
        
        # Query ChromaDB
        query_results = self.collection.query(
            query_embeddings=[features.tolist()],
            n_results=k
        )
        
        # Organize results
        results = []
        seen_product_ids = set()  # For deduplication
        
        if not query_results or 'ids' not in query_results or not query_results['ids']:
            return []
        
        for i, item_id in enumerate(query_results['ids'][0]):
            metadata = query_results['metadatas'][0][i]
            distance = query_results['distances'][0][i] if 'distances' in query_results else 0.0
            
            product_id = metadata.get('product_id')
            
            # Skip if we already added this product (deduplication)
            if product_id in seen_product_ids:
                continue
            
            seen_product_ids.add(product_id)
            
            # Calculate similarity score from distance (inverted)
            similarity = 1.0 / (1.0 + distance)
            
            results.append({
                'product_id': product_id,
                'image_id': metadata.get('image_id'),
                'category_id': metadata.get('category_id'),
                'subcategory_id': metadata.get('subcategory_id'),
                'similarity': float(similarity),
                'distance': float(distance)
            })
        
        return results
    
    def get_product_images(self, product_id: str):
        """
        Get all images for a specific product
        
        Args:
            product_id: Unique identifier for the product
        
        Returns:
            List of image metadata
        """
        if product_id not in self.product_map:
            return []
        
        # Get all image IDs for this product
        image_ids = self.product_map[product_id]
        
        # Query ChromaDB for these images
        try:
            result = self.collection.get(
                ids=image_ids,
                include=["metadatas"]
            )
            
            images = []
            for metadata in result.get('metadatas', []):
                images.append({
                    'image_id': metadata.get('image_id'),
                    'image_url': metadata.get('image_url')
                })
            
            return images
        except Exception as e:
            print(f"Error getting images for product {product_id}: {e}")
            return []
    def batch_insert_products(self, products_data: List[Dict[str, Any]]):
        """
        Insert multiple products in batch mode for better efficiency
        
        Args:
            products_data: List of product dictionaries with product_id, images, category_id, subcategory_id
            
        Returns:
            List of results for each product
        """
        results = []
        
        # Collect all image IDs across all products for a single lookup
        all_image_ids = []
        image_product_map = {}  # Maps image_id to product_id
        
        # First pass: collect all image IDs
        for product in products_data:
            product_id = product.get('product_id')
            images = product.get('images', [])
            
            for image in images:
                image_id = image.get('image_id')
                if image_id:
                    all_image_ids.append(image_id)
                    image_product_map[image_id] = product_id
        
        # Query ChromaDB only once for all image IDs
        existing_ids = set()
        if all_image_ids:
            try:
                existing_results = self.collection.get(
                    ids=all_image_ids,
                    include=["metadatas"]
                )
                existing_ids = set(existing_results.get('ids', []))
                print(f"Found {len(existing_ids)} already existing images out of {len(all_image_ids)} in batch")
            except Exception as e:
                print(f"Error checking existing images: {e}")
        
        # Second pass: process each product
        for product in products_data:
            product_id = product.get('product_id')
            category_id = product.get('category_id')
            subcategory_id = product.get('subcategory_id')
            images = product.get('images', [])
            
            # Process this product
            success, num_added, failed = self._process_product_batch(
                product_id, 
                images, 
                category_id, 
                subcategory_id, 
                existing_ids
            )
            
            results.append({
                'product_id': product_id,
                'success': success,
                'images_added': num_added,
                'failed_images': failed
            })
        
        return results

    def _process_product_batch(self, product_id, images, category_id, subcategory_id, existing_ids):
        """Helper method to process a single product during batch insert"""
        num_added = 0
        failed_images = []
        
        for image in images:
            # Extract image info
            image_id = image.get('image_id')
            image_url = image.get('url')
            
            if not image_id or not image_url:
                continue
            
            # Skip if already in database
            if image_id in existing_ids:
                # Make sure the product map is updated
                if product_id not in self.product_map:
                    self.product_map[product_id] = []
                if image_id not in self.product_map[product_id]:
                    self.product_map[product_id].append(image_id)
                continue
            
            try:
                # Extract features
                features = self.feature_extractor.extract_from_url(image_url)
                if features is None:
                    failed_images.append(image_id)
                    continue
                
                # Prepare metadata
                metadata = {
                    'product_id': product_id,
                    'image_id': image_id,
                    'category_id': category_id if category_id else '',
                    'subcategory_id': subcategory_id if subcategory_id else '',
                    'image_url': image_url
                }
                
                # Add to ChromaDB
                self.collection.upsert(
                    ids=[image_id],
                    embeddings=[features.tolist()],
                    metadatas=[metadata]
                )
                
                # Update mappings
                if product_id not in self.product_map:
                    self.product_map[product_id] = []
                if image_id not in self.product_map[product_id]:
                    self.product_map[product_id].append(image_id)
                
                num_added += 1
                
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
                failed_images.append(image_id)
        
        return (num_added > 0, num_added, failed_images)
    def get_stats(self):
        """
        Get statistics about the database
        
        Returns:
            Dictionary with stats
        """
        # Count items in collection
        all_items = self.collection.get()
        total_images = len(all_items.get('ids', []))
        
        return {
            'total_products': len(self.product_map),
            'total_images': total_images,
            'model_name': self.model_name
        }