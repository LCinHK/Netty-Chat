import os
# Disable ChromaDB Telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress logging from libraries that might be noisy
import logging
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

# Fix for Posthog Telemetry Error
try:
    import posthog
    original_capture = posthog.capture
    def mocked_capture(distinct_id, event, properties=None, groups=None, send_feature_flags=False):
        pass
    posthog.capture = mocked_capture
except ImportError:
    pass

import chromadb
import shutil
import os

PATH_VECTOR_DB = "./arag/chromaVectorStore"

print(f"Cleaning up {PATH_VECTOR_DB}...")

# Option 1: Try to reset via client (if API allows)
try:
    client = chromadb.PersistentClient(path=PATH_VECTOR_DB)
    cols = client.list_collections()
    for c in cols:
        print(f"Deleting collection: {c.name}")
        client.delete_collection(c.name)
except Exception as e:
    print(f"Error deleting collections: {e}")

print("Done.")
