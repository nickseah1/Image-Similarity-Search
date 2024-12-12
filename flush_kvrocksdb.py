# flush_kvrocksdb.py

from kvrocksdb_handler import KVRocksDBHandler

def flush_db():
    kv_handler = KVRocksDBHandler()
    kv_handler.client.flushdb()
    print("KVRocksDB flushed successfully.")

if __name__ == "__main__":
    flush_db()
