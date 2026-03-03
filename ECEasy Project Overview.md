# ECEasy Project Overview

## What It Is
ECEasy is a **RAG-powered chatbot** designed for HKUST ECE (Electronic & Computer Engineering) students. It answers questions about ECE courses, program requirements, and university life by combining a local knowledge base with web search.

---

## Architecture

### Backend (`eceasy_local_server.py`)
- Built with **FastAPI**, served at `http://localhost:8000`
- Supports three LLM providers (configured via `.env`):
  - **Ollama** (local, default — currently `qwen3:4b`)
  - **OpenAI** (e.g. `gpt-4o`)
  - **DeepSeek** (e.g. `deepseek-chat`)
- Query pipeline per request:
  1. **RAG retrieval** via `arag/arag.py` → ChromaDB (Chroma vector store)
  2. **Web search fallback** via DuckDuckGo (`ddgs`) if RAG yields < 8 results
  3. **LLM streaming response** with citations
  4. **Related questions generation** (second LLM call)
  5. Results cached to a `shelve` KV store (`.kv` files)
- Streaming protocol: `[contexts JSON]\n\n__LLM_RESPONSE__\n\n[LLM text]\n\n__RELATED_QUESTIONS__\n\n[questions JSON]`

### RAG (`arag/arag.py`)
- Uses **ChromaDB** as the vector store (`./arag/chromaVectorStore/`)
- Embedding model: `sentence-transformers/all-mpnet-base-v2` (HuggingFace, cached in `./arag/modelCache/`)
- Collection name: `"nettyRAG"` ⚠️ — **this is the old Netty/computer-networks collection, NOT ECE content**
- Cosine distance threshold (`MAGIC_NUMBER = 1.0`) — documents with score ≥ 1.0 are filtered out

### Ingestion Scripts
| Script | Target Vector Store | Purpose |
|---|---|---|
| `ingest_local_data.py` | ChromaDB (`./arag/chromaVectorStore/`, collection `nettyRAG`) | Ingest from `./localData/` (PDFs & TXTs) |
| `ingest_university.py` | **FAISS** (`./faiss_index_university/`) | Ingest from `./knowledge/university_life/` (old path, unused) |

> ⚠️ **`ingest_university.py` uses FAISS** but the server never reads from the FAISS index — it only reads from ChromaDB.

### Knowledge Base (`ECEknowledge/`)
Rich collection of ECE-relevant documents ready to be ingested:
- **PDFs**: BEng ECE program overview, BEng MEIC overview, HKUST Common Core Program
- **Course syllabi**: ELEC (34 courses), COMP (22 courses), MATH (19 courses), PHYS (2 courses)
- **Program requirements**: `25-26elec.pdf`, `25-26meic.pdf`, `minor-robo.pdf`
- **FAQ**: `FAQs.docx`

> ⚠️ None of this has been ingested yet — the current ChromaDB contains computer networking data from the old Netty project.

### Frontend
Two UIs exist:
1. **`web/`** — Full Next.js/React/TypeScript frontend (with Tailwind CSS, Mermaid diagrams, KaTeX math rendering, syntax highlighting, citation popovers). Built output served from `ui/` by FastAPI.
2. **`ui/`** — Pre-built static output (HTML/JS) already served by FastAPI at `/ui/index.html`

The UI streams responses and parses the three-part protocol, rendering:
- **Sources panel** (with favicons, URLs, page numbers)
- **Markdown answer** (with inline citation badges, Mermaid diagrams, math)
- **Related questions** panel

### Prompts (`ecEasyPrompts.py`)
- `_rag_query_text`: System prompt that instructs the LLM to answer using `[[citation:x]]` references and optionally generate a Mermaid diagram
- `_more_questions_prompt`: Prompt for generating 3 related follow-up questions

---

## Current State & Key Gaps

| Issue | Detail |
|---|---|
| ❌ Wrong RAG database | `arag/chromaVectorStore/` contains computer networking content (old Netty project), not ECE content |
| ❌ ECEknowledge not ingested | All the PDFs/DOCX in `ECEknowledge/` need to be ingested into a new vector store |
| ⚠️ Collection name mismatch | Still named `"nettyRAG"` — should be updated for ECEasy |
| ⚠️ `ingest_university.py` misaligned | Uses FAISS and wrong data path; server never reads from it |
| ⚠️ Preset queries are old | `page.tsx` still has computer networking sample questions |
| ⚠️ `localData/` is empty | Currently empty; the legacy data is in `legacy/localData/` |

---

## Configuration (`.env`)
Provider selection and API keys are set via `.env` (gitignored). An `.env.example` exists for reference.

---

## How to Run
```
.\run_local_server.bat
# or
pip install -r requirements_local.txt
python eceasy_local_server.py
```
Server → `http://localhost:8000/ui/index.html`

