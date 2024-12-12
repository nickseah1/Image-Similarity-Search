# main.py

import sys
import os
import logging
import hashlib
import numpy as np
import faiss

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from embedding_extractor import EmbeddingExtractor
from data_loader import DataLoader
from kvrocksdb_handler import KVRocksDBHandler

class FaissHandlerWithKVRocks:
    def __init__(self, dim, kv_handler):
        self.dim = dim
        self.kv_handler = kv_handler
        self.index = faiss.IndexFlatL2(dim)  # Example index type, can be changed based on your requirements

    def build_index_from_kv(self):
        # Retrieve all embeddings from KVRocksDB
        embeddings = []
        idx = 0
        while True:
            embedding = self.kv_handler.retrieve_embedding(f"embedding_vector_{idx}")
            if embedding is None:
                break
            embeddings.append(embedding)
            idx += 1

        if embeddings:
            # Convert list of embeddings to numpy array
            embeddings_np = np.array(embeddings).astype('float32')
            # Add embeddings to the FAISS index
            self.index.add(embeddings_np)
        else:
            raise ValueError("No embeddings found in KVRocksDB to build FAISS index.")

    def search(self, query_embedding, k):
        query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        return indices[0], distances[0]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_image_hash(image_path):
    """Compute MD5 hash of the image for duplicate detection."""
    hash_md5 = hashlib.md5()
    try:
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {image_path}: {e}")
        return None

def main():
    dim = 2048  # Dimension of embeddings
    extractor = EmbeddingExtractor()
    data_loader = DataLoader()
    kv_handler = KVRocksDBHandler()
    faiss_handler = FaissHandlerWithKVRocks(dim, kv_handler)

    # Load images and extract embeddings
    logger.info("Loading images and extracting embeddings...")
    image_paths = data_loader.load_images("data/custom_images/")
    for idx, image_path in enumerate(image_paths):
        # Compute hash to check for duplicates
        image_hash = compute_image_hash(image_path)
        if image_hash is None:
            continue  # Skip if hash computation failed

        # Check if the image hash already exists
        existing_key = kv_handler.retrieve_metadata(f"hash_{image_hash}")
        if existing_key:
            logger.info(f"Duplicate image found. Skipping: {image_path}")
            continue  # Skip duplicate images

        # Extract embedding for each image
        embedding = extractor.extract(image_path)
        if embedding is None or len(embedding) != dim:
            logger.warning(f"Failed to extract embedding for {image_path}. Skipping.")
            continue

        key = f"vector_{idx}"
        # Store embeddings and metadata in KVRocks
        kv_handler.store_embedding(f"embedding_vector_{idx}", embedding)
        kv_handler.store_metadata(f"metadata_vector_{idx}", image_path)
        kv_handler.store_metadata(f"hash_{image_hash}", key)
        logger.info(f"Stored embedding and metadata for {key}: {image_path}")

    # Build FAISS index from KVRocks embeddings
    logger.info("Building FAISS index...")
    faiss_handler.build_index_from_kv()

    # Search example
    query_image = "data/custom_images/query.jpg"  # Path to query image
    if not os.path.exists(query_image):
        logger.error(f"Query image not found: {query_image}")
        return

    query_embedding = extractor.extract(query_image)
    if query_embedding is None or len(query_embedding) != dim:
        logger.error("Invalid query embedding.")
        return

    similar_keys, similar_distances = faiss_handler.search(query_embedding, k=5)

    # Retrieve and display results
    logger.info("Query Results:")
    for key, dist in zip(similar_keys, similar_distances):
        metadata_key = f"metadata_vector_{key}"  # Ensure consistent key format
        image_path = kv_handler.retrieve_metadata(metadata_key)
        logger.info(f"Retrieved metadata for {metadata_key}: {image_path}")
        if image_path:
            logger.info(f"Image: {image_path}, Distance: {dist}")
        else:
            logger.warning(f"Metadata for key {metadata_key} not found.")

if __name__ == "__main__":
    main()

