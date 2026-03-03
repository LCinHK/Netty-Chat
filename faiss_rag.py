"""
FAISS-based RAG module for ECEasy.
Uses the FAISS index built from ECEknowledge/ (by ingest_university.py).
Embedding model: all-MiniLM-L6-v2  (matches the model used during ingestion)
"""

import os
import re
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ======== Configuration ========
FAISS_INDEX_PATH = "faiss_index_university"
FAISS_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Similarity score threshold.
# FAISS returns L2 distance by default; lower is more similar.
# Chunks with distance >= this value are considered irrelevant and filtered out.
FAISS_SCORE_THRESHOLD = 1.5

# ======== Load embedding model & vector store once at module import ========
_embeddings = HuggingFaceEmbeddings(
    model_name=FAISS_EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

_vectorstore = None
try:
    _vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings=_embeddings,
        allow_dangerous_deserialization=True,  # Required for local pickled index
    )
    logger.info(f"[FAISS RAG] Index loaded successfully from '{FAISS_INDEX_PATH}'")
    print(f"[INFO] FAISS RAG index loaded from '{FAISS_INDEX_PATH}'")
except Exception as e:
    logger.warning(f"[FAISS RAG] Could not load FAISS index: {e}. FAISS RAG will be disabled.")
    print(f"[WARNING] Could not load FAISS index: {e}. FAISS RAG will be disabled.")


def get_rag_context(query: str):
    """
    Retrieve relevant document chunks from the FAISS index for the given query.
    Returns a list of context dicts compatible with the server's streaming pipeline:
        [{ 'name': str, 'snippet': str, 'url': str }, ...]
    """
    if _vectorstore is None:
        return []

    try:
        # similarity_search_with_score returns (Document, score) tuples
        retrieved = _vectorstore.similarity_search_with_score(query, k=8)
    except Exception as e:
        logger.error(f"[FAISS RAG] Search failed: {e}")
        return []

    # Debug
    print(f"[FAISS RAG Debug] Query: {query}")
    for doc, score in retrieved:
        print(f"[FAISS RAG Debug] score={score:.4f} | {doc.page_content[:60]}...")

    context = []
    for doc, score in retrieved:
        if score >= FAISS_SCORE_THRESHOLD:
            continue  # Too dissimilar — skip

        metadata = doc.metadata
        # LangChain PDF loaders store the source path in 'source'
        file_path = metadata.get("source", metadata.get("file_path", ""))

        if "page" in metadata:
            # page is 0-indexed in LangChain loaders
            page_num = int(metadata["page"]) + 1
            name = f"Page {page_num}, {os.path.basename(file_path)}"
            url_suffix = f"#page={page_num}"
        else:
            name = os.path.basename(file_path) if file_path else "Source"
            url_suffix = ""

        # Normalise path separators and strip leading ../
        clean_url_path = re.sub(r"\.\.", "", re.sub(r"\\+", "/", file_path))

        context.append({
            "name": name,
            "snippet": doc.page_content,
            "url": clean_url_path + url_suffix,
        })

    # De-duplicate by snippet content
    unique_context = list({entry["snippet"]: entry for entry in context}.values())

    return unique_context

