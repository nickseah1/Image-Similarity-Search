# kvrocksdb_handler.py

import redis
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KVRocksDBHandler:
    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def store_embedding(self, key, embedding):
        """Store a vector embedding."""
        embedding_str = json.dumps(embedding.tolist())
        self.client.set(key, embedding_str)

    def retrieve_embedding(self, key):
        """Retrieve a vector embedding."""
        embedding_str = self.client.get(key)
        if embedding_str:
            return np.array(json.loads(embedding_str))
        return None

    def store_metadata(self, key, metadata):
        """Store associated metadata like image paths."""
        if not self.client.exists(key):
            self.client.set(key, metadata)
        else:
            logger.warning(f"Key {key} already exists. Skipping.")


    def retrieve_metadata(self, key):
        """Retrieve metadata associated with a key."""
        metadata = self.client.get(key)
        if metadata:
            return metadata
        return None

    def delete_key(self, key):
        """Delete a key from the database."""
        self.client.delete(key)

    def get_all_keys(self, pattern="*"):
        """Retrieve all keys matching a pattern."""
        return self.client.keys(pattern)
