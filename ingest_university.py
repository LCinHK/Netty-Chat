"""
Ingestion script for ECEasy FAISS knowledge base.
Reads all .pdf, .docx, and .txt files from ECEknowledge/ and builds
(or rebuilds) the FAISS index at faiss_index_all-MiniLM-L6-v2/.

Dependencies (install before running):
    pip install pypdf docx2txt faiss-cpu langchain-community langchain-huggingface langchain-text-splitters
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ======== Configuration ========
DATA_PATH = Path("ECEknowledge")       # Source knowledge folder
# Note: INDEX_PATH is now derived automatically from EMBEDDING_MODEL_HUB_NAME in .env
# e.g. "BAAI/bge-small-en-v1.5" → "./faiss_index_bge-small-en-v1.5"

# Files / patterns to skip (e.g. macOS metadata files)
SKIP_PATTERNS = {".DS_Store"}


def _index_name_from_hub(hub_name: str) -> str:
    """
    Derives a filesystem-safe FAISS index folder name from a Hub model name.
      "BAAI/bge-small-en-v1.5"  →  "faiss_index_bge-small-en-v1.5"
      "all-MiniLM-L6-v2"        →  "faiss_index_all-MiniLM-L6-v2"
    """
    short = hub_name.split("/")[-1]
    return f"faiss_index_{short}"


def _resolve_embedding_model() -> tuple[str, str]:
    """
    Returns (model_name_or_path, faiss_index_path).

    Model resolution priority:
      1. If EMBEDDING_MODEL_LOCAL_PATH in .env points to an existing local directory
         → use it directly (fully offline; sets TRANSFORMERS_OFFLINE=1).
      2. Otherwise use EMBEDDING_MODEL_HUB_NAME as a Hub ID for auto-download/cache.

    The FAISS index folder is always derived from EMBEDDING_MODEL_HUB_NAME so that
    different models store their indexes in separate directories and never overwrite
    each other. Must stay in sync with faiss_rag.py.
    """
    hub_name = os.environ.get("EMBEDDING_MODEL_HUB_NAME", "all-MiniLM-L6-v2").strip()
    index_path = _index_name_from_hub(hub_name)

    local_path = os.environ.get("EMBEDDING_MODEL_LOCAL_PATH", "").strip()
    if local_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.normpath(os.path.join(base_dir, local_path))
        if os.path.isdir(resolved):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            print(f"[Embedding] Using local model folder: '{resolved}' (offline)")
            print(f"[Embedding] FAISS index will be saved to: '{index_path}'")
            return resolved, index_path
        else:
            print(f"[Embedding] WARNING: EMBEDDING_MODEL_LOCAL_PATH '{local_path}' "
                  f"(resolved: '{resolved}') not found — falling back to HuggingFace Hub.")

    print(f"[Embedding] Using HuggingFace Hub model: '{hub_name}' (requires internet on first run)")
    print(f"[Embedding] FAISS index will be saved to: '{index_path}'")
    return hub_name, index_path


def load_all_documents(data_path: Path):
    """
    Load all .pdf, .docx, and .txt files recursively from data_path.
    Uses per-file loaders for better error isolation and metadata accuracy.
    """
    all_docs = []
    skipped = []

    # --- PDFs (per-file for accurate page metadata) ---
    pdf_files = sorted(data_path.rglob("*.pdf"))
    print(f"  Found {len(pdf_files)} PDF file(s)")
    for pdf_path in pdf_files:
        if pdf_path.name in SKIP_PATTERNS:
            continue
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Ensure 'source' metadata is the relative path string for consistency
            for doc in docs:
                doc.metadata["source"] = str(pdf_path)
            all_docs.extend(docs)
            print(f"    [PDF]  {pdf_path.relative_to(data_path)}  ({len(docs)} pages)")
        except Exception as e:
            skipped.append((str(pdf_path), str(e)))
            print(f"    [PDF]  SKIP {pdf_path.name}: {e}")

    # --- DOCX (per-file) ---
    docx_files = sorted(data_path.rglob("*.docx"))
    print(f"  Found {len(docx_files)} DOCX file(s)")
    for docx_path in docx_files:
        if docx_path.name in SKIP_PATTERNS:
            continue
        try:
            loader = Docx2txtLoader(str(docx_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(docx_path)
            all_docs.extend(docs)
            print(f"    [DOCX] {docx_path.relative_to(data_path)}  ({len(docs)} doc(s))")
        except Exception as e:
            skipped.append((str(docx_path), str(e)))
            print(f"    [DOCX] SKIP {docx_path.name}: {e}")

    # --- TXT (per-file) ---
    txt_files = sorted(data_path.rglob("*.txt"))
    print(f"  Found {len(txt_files)} TXT file(s)")
    for txt_path in txt_files:
        if txt_path.name in SKIP_PATTERNS:
            continue
        try:
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs = loader.load()
            except Exception:
                loader = TextLoader(str(txt_path), encoding="latin-1")
                docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(txt_path)
            all_docs.extend(docs)
            print(f"    [TXT]  {txt_path.relative_to(data_path)}  ({len(docs)} doc(s))")
        except Exception as e:
            skipped.append((str(txt_path), str(e)))
            print(f"    [TXT]  SKIP {txt_path.name}: {e}")

    if skipped:
        print(f"\n  Warning: {len(skipped)} file(s) could not be loaded:")
        for path, err in skipped:
            print(f"    - {path}: {err}")

    return all_docs


def main():
    if not DATA_PATH.exists() or not any(DATA_PATH.iterdir()):
        print(f"Error: Folder '{DATA_PATH}' is empty or doesn't exist.")
        print("→ Create it and add at least one .txt / .docx / .pdf file")
        return

    print(f"Loading documents from '{DATA_PATH}'...")
    docs = load_all_documents(DATA_PATH)
    print(f"\nTotal raw pages/docs loaded: {len(docs)}")

    if len(docs) == 0:
        print("No documents loaded → nothing to index. Add files and retry.")
        return

    # Split into chunks
    print("\nSplitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Resolve embedding model and derived index path
    model_name, index_path = _resolve_embedding_model()

    # Embed and store in FAISS
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},      # Change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding and building FAISS index (this may take a while)...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Remove old index for this model before saving to avoid stale data
    old_index = Path(index_path)
    if old_index.exists():
        shutil.rmtree(old_index)
        print(f"Removed old index at '{index_path}'")

    vectorstore.save_local(index_path)
    print(f"\nDone! FAISS index saved to: '{index_path}' ({len(chunks)} vectors)")


if __name__ == "__main__":
    main()