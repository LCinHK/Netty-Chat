import os
# Disable ChromaDB Telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress logging from libraries that might be noisy
import logging
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

# Fix for Posthog Telemetry Error: "capture() takes 1 positional argument but 3 were given"
# This patches the posthog library to ignore extra arguments during capture calls from chromadb
try:
    import posthog
    original_capture = posthog.capture
    def mocked_capture(distinct_id, event, properties=None, groups=None, send_feature_flags=False):
        # We just swallow the call to avoid the error
        pass
    posthog.capture = mocked_capture
except ImportError:
    pass

import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration matching arag/arag.py
PATH_MODEL_CACHE = "./arag/modelCache"
PATH_VECTOR_DB = "./arag/chromaVectorStore"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "nettyRAG"
DATA_DIR = "./localData"

def ingest():
    print(f"Initializing embedding model: {EMBEDDING_MODEL}...")
    print(f"Model will be cached at: {os.path.abspath(PATH_MODEL_CACHE)}")
    print("This allows the model to be downloaded once and reused offline (mostly).")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=PATH_MODEL_CACHE,
    )

    print(f"Loading documents from {DATA_DIR}...")
    documents = []

    # 1. Load PDFs (Limited to speed up)
    # We use glob to find all pdfs recursively
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True)
    print(f"Found {len(pdf_files)} PDF files.")

    # Filter to prioritize Standards or limit count
    # For now, let's limit to 5 PDFs to save time, unless user wants full ingestion later.
    pdf_files = pdf_files[:5]

    for pdf_path in pdf_files:
        try:
            print(f"Loading {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

    # 2. Load Text files (RFCs in Standards folder)
    txt_files = glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
    print(f"Found {len(txt_files)} TXT files.")

    for txt_path in txt_files:
        try:
            # print(f"Loading {txt_path}...") # Reduce noise if many files
            # RFCs might be utf-8 or ascii. Try autodetectish or simple retry
            try:
                loader = TextLoader(txt_path, encoding='utf-8')
                docs = loader.load()
            except Exception:
                loader = TextLoader(txt_path, encoding='latin-1')
                docs = loader.load()

            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {txt_path}: {e}")

    print(f"Total raw documents loaded: {len(documents)}")

    if not documents:
        print("No documents found. Exiting.")
        return

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Ingesting into ChromaDB at {PATH_VECTOR_DB} (Collection: {COLLECTION_NAME})...")
    # Initialize Chroma
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PATH_VECTOR_DB
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Adding batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}...")
        vectorstore.add_documents(batch)

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
