import os
import ray
import chromadb
import time  # Import time module
from PIL import Image
from typing import List, Tuple
from src.feature_extractor import FeatureExtractor

# Initialize Ray
ray.init(ignore_reinit_error=True,memory=2 * 1024 * 1024 * 1024)

# Global feature extractor
extractor = FeatureExtractor()


@ray.remote
def process_image_ray(img_path: str) -> Tuple[str, List[float]]:
    """Extract features from an image using Ray (runs in parallel)."""
    try:
        image = Image.open(img_path).convert('RGB')
        feature_vector = extractor.extract(image).tolist()
        return img_path, feature_vector
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


class ImageDatabase:
    def __init__(self, db_path="chroma_full", dataset_dir="archive", static_dir="static"):
        """Initialize ChromaDB and set up directories."""
        self.dataset_dir = dataset_dir
        self.static_dir = static_dir
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="image_vectors")
    def clear_static_directory(self,static_dir):
        for filename in os.listdir(static_dir):
            file_path = os.path.join(static_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    def get_existing_images(self) -> set:
        """Retrieve already indexed images from ChromaDB."""
        existing_records = self.collection.get()
        return set(record["path"] for record in existing_records["metadatas"]) if "metadatas" in existing_records else set()

    def insert_new_images(self, batch_size=100):
        """Scan dataset directory and insert new images in parallel using Ray."""
        start_time = time.time()  # Start total timer

        existing_images = self.get_existing_images()
        all_images = [os.path.join(root, file) for root, _, files in os.walk(self.dataset_dir)
                      for file in files if file.endswith(('png', 'jpg', 'jpeg'))]

        # Filter only new images
        new_images = [img for img in all_images if img not in existing_images]

        if not new_images:
            print("No new images to insert.")
            return

        print(f"Total new images: {len(new_images)}. Processing in batches of {batch_size}...")

        total_images = len(new_images)
        processed_images = 0

        for i in range(0, total_images, batch_size):
            batch_start_time = time.time()  # Start batch timer
            batch = new_images[i:i + batch_size]

            # Submit tasks in parallel using Ray
            futures = [process_image_ray.remote(img) for img in batch]
            results = ray.get(futures)  # Fetch all results in parallel

            # Remove failed extractions (None values)
            results = [res for res in results if res is not None]

            if results:
                image_ids = [f"img_{os.path.basename(img_path)}" for img_path, _ in results]
                embeddings = [features for _, features in results]
                metadatas = [{"path": img_path} for img_path, _ in results]

                self.collection.add(ids=image_ids, embeddings=embeddings, metadatas=metadatas)

                processed_images += len(results)
                batch_end_time = time.time()  # End batch timer
                batch_time = batch_end_time - batch_start_time
                print(f"Processed {len(results)} images in {batch_time:.2f} seconds. Total processed: {processed_images}/{total_images}")

        total_time = time.time() - start_time  # End total timer
        avg_time_per_image = total_time / total_images if total_images > 0 else 0
        print(f"✅ Finished processing {total_images} images in {total_time:.2f} seconds.")
        print(f"⏳ Average processing time per image: {avg_time_per_image:.3f} seconds.")
