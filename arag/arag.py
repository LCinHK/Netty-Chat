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

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
import torch

from dotenv import load_dotenv
load_dotenv(override = True) 

PATH_MODEL_CACHE = "./arag/modelCache"
PATH_VECTOR_DB = "./arag/chromaVectorStore"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

MAGIC_NUMBER = 1.0
# Cosine distance threshold, if content below this threshold,
# it is considered as similar and therefore used as context

GPU_available = torch.cuda.is_available()

# Check if GPU is available
if GPU_available:
    print("[INFO] GPU is available")
else:
    print("[INFO] GPU is not available. Using CPU instead")

embeddingModel = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL,
    model_kwargs = { "device": "cuda" if GPU_available else "cpu", "trust_remote_code": True },
    encode_kwargs = { "normalize_embeddings": True },
    cache_folder = PATH_MODEL_CACHE,
)

chromaVectorStore = Chroma(
    collection_name = "nettyRAG",
    embedding_function = embeddingModel,
    persist_directory = PATH_VECTOR_DB
)
chromaVectorStoreRetriever = chromaVectorStore.as_retriever()

def get_rag_context(query: str):
    retrieved_docs = chromaVectorStore.similarity_search_with_score(query)

    # Debug info for scores
    print(f"[RAG Debug] Query: {query}")
    for doc, score in retrieved_docs:
        print(f"[RAG Debug] Found doc: {doc.page_content[:50]}... | Score: {score}")

    context = []
    for doc_tuple in retrieved_docs:
        doc, score = doc_tuple
        if score >= MAGIC_NUMBER:
            continue

        metadata = doc.metadata
        # Fail-safe: Use 'file_path' if exists, else 'source', else empty string
        file_path = metadata.get('file_path', metadata.get('source', ''))

        if 'page' in metadata:
            name = f"Page {metadata['page'] + 1}, {os.path.basename(file_path)}"
            url_path = file_path
            url_suffix = f"#page={metadata['page'] + 1}"
        else:
            name = os.path.basename(file_path)
            url_path = file_path
            url_suffix = ""

        # Normalize path separators and remove .. for security/formatting
        clean_url_path = re.sub(r'\.\.', '', re.sub(r"\\?\\", "/", url_path))

        context.append({
            'name': name,
            'snippet': doc.page_content,
            'url': clean_url_path + url_suffix
        })

    unique_context = {entry['snippet']: entry for entry in context}.values()
    unique_context_list = list(unique_context)

    return unique_context_list
