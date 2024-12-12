import faiss
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaissHandlerWithKVRocks:
    def __init__(self, dim, kv_handler, index_path="faiss.index"):
        self.dim = dim
        self.kv_handler = kv_handler
        self.index_path = index_path
        self.index_to_key = []
        self.index = faiss.IndexFlatIP(dim)  # Inner Product index for cosine similarity

        # Attempt to load existing index or build a new one
        if os.path.exists(self.index_path):
            try:
                self.load_index()
                logger.info(f"FAISS index loaded from {self.index_path}.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Rebuilding index...")
                self.build_index_from_kv()
        else:
            logger.info("FAISS index not found. Building from KVRocksDB...")
            self.build_index_from_kv()

    def normalize_embeddings(self, embeddings):
        """Normalize embeddings to unit vectors."""
        faiss.normalize_L2(embeddings)
        return embeddings

    def hybrid_search(self, query_vector, k=5):
        """Perform hybrid search using FAISS and KVRocksDB."""
        query_vector = np.array([query_vector], dtype="float32")
        query_vector = self.normalize_embeddings(query_vector)

        # Perform initial FAISS search
        distances, indices = self.index.search(query_vector, k)
        similar_keys = []
        similar_distances = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.index_to_key):
                similar_keys.append(self.index_to_key[idx])
                similar_distances.append(dist)

        # If results are insufficient, fetch from KVRocksDB
        if len(similar_keys) < k:
            logger.info("Insufficient neighbors found. Fetching additional vectors from KVRocksDB...")
            self.rebuild_index_from_kv()
            return self.hybrid_search(query_vector, k)  # Retry after rebuilding the index

        return similar_keys, similar_distances

    def rebuild_index_from_kv(self):
        """Rebuild FAISS index with vectors from KVRocksDB."""
        logger.info("Rebuilding FAISS index...")
        self.index.reset()
        self.index_to_key = []

        keys = self.kv_handler.get_all_keys("embedding_vector_*")
        embeddings = []

        for key in keys:
            embedding = self.kv_handler.retrieve_embedding(key)
            if embedding is not None:
                embeddings.append(embedding)
                self.index_to_key.append(key.replace("embedding_vector_", "vector_"))

        if embeddings:
            embeddings_np = np.array(embeddings, dtype="float32")
            embeddings_np = self.normalize_embeddings(embeddings_np)
            self.index.add(embeddings_np)
            self.save_index()
            logger.info(f"FAISS index rebuilt with {len(embeddings)} vectors.")
        else:
            logger.warning("No embeddings found in KVRocksDB to rebuild the FAISS index.")

    def save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, self.index_path)
        logger.info(f"FAISS index saved to {self.index_path}.")

    def load_index(self):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(self.index_path)
        keys = self.kv_handler.get_all_keys("embedding_vector_*")
        self.index_to_key = [key.replace("embedding_vector_", "vector_") for key in keys]
