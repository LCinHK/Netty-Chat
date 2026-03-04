"""
This module implements a local-only backend server for ECEasy.
It replaces the Lepton AI dependency with local Ollama, OpenAI, or DeepSeek as the LLM provider.
"""

import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Disable ChromaDB Telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress logging from libraries that might be noisy
import logging
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import warnings
# Suppress Pydantic V2 deprecation warnings coming from ChromaDB
warnings.filterwarnings("ignore", message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*")

import re
import threading
import shelve
import uuid
from typing import List, Generator, Optional
from pydantic import BaseModel

# ======== FastAPI Imports ========
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx
from loguru import logger

# ======== OpenAI / Ollama Imports ========
import openai

# ======== Search Engine Functions ========
try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
except ImportError:
    logger.warning("duckduckgo_search / ddgs not installed. Web search will be disabled.")
    DDGS = None

# ======== Local Imports ========
import ecEasyPrompts
try:
    from arag.arag import get_rag_context
except ImportError:
    logger.warning("Could not import arag.arag. RAG functionality will be disabled.")
    def get_rag_context(query): return []

# ======== Configuration ========
from dotenv import load_dotenv
load_dotenv(override=True)

# --- LLM Provider Selection ---
# Set LLM_PROVIDER to "ollama", "openai", or "deepseek"
# You can set this in your .env file or default here.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

# --- Common Config ---
KV_NAME = "eceasy-chat-local.kv"
REFERENCE_COUNT = 8 # change from 8 to 0
SHOULD_DO_RELATED_QUESTIONS = False # change from true to false

# --- Provider Specific Config ---

# 1. Ollama Configuration
# URL where your local Ollama instance is running.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# The model name in Ollama (e.g., "qwen2.5:14b").
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

# 2. OpenAI Configuration
# Your OpenAI API Key.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_REMOTE_OPENAI_API_KEY", ""))
# The model to use.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", os.environ.get("LLM_REMOTE_OPENAI_MODEL", "gpt-4o"))
# Custom Base URL for OpenAI-compatible endpoints (e.g. for proxies or local surrogates)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", os.environ.get("LLM_REMOTE_OPENAI_URL", "https://api.openai.com/v1"))

# 3. DeepSeek Configuration
# Your DeepSeek API Key.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("LLM_REMOTE_API_KEY", ""))
# DeepSeek API Base URL.
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("LLM_REMOTE_URL", "https://api.deepseek.com"))
# The model to use.
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_REMOTE_MODEL", "deepseek-chat"))

# Helper to get the current model name based on provider
def get_current_model_name():
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    elif LLM_PROVIDER == "deepseek":
        return DEEPSEEK_MODEL
    return OLLAMA_MODEL  # Default to Ollama

LLM_MODEL = get_current_model_name()
logger.info(f"Using LLM Provider: {LLM_PROVIDER}, Model: {LLM_MODEL}")

# Stop words for the LLM
# OpenAI API limits to 4 stop sequences.
STOP_WORDS = [
    "<|im_end|>",
    "[End]",
    "\nReferences:\n",
    "\nSources:\n",
]

# ======== FAISS University Knowledge Base ========
FAISS_INDEX_PATH = "faiss_index_university"

# Load embeddings once
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vector store once at startup
vectorstore = None
try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Required for local pickled index
    )
    logger.info(f"FAISS university index loaded successfully from {FAISS_INDEX_PATH}")
except Exception as e:
    logger.warning(f"Could not load FAISS index: {e}. University RAG will be disabled.")

# ======== Models ========

class QueryRequest(BaseModel):
    query: str
    search_uuid: str
    generate_related_questions: Optional[bool] = True

# ======== Helper Functions ========

def get_llm_client():
    """
    Returns a thread-local OpenAI client configured for the selected provider.
    """
    thread_local = threading.local()
    if hasattr(thread_local, "client"):
        return thread_local.client

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is missing. Please set it in .env or environment variables.")
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

    elif LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            logger.error("DEEPSEEK_API_KEY is missing. Please set it in .env or environment variables.")
        client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )

    else: # Default to Ollama
        client = openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama", # API key is not required for local Ollama
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        )

    thread_local.client = client
    return client

def search_with_duckduckgo(query: str) -> List[dict]:
    """
    Search using DuckDuckGo (via ddgs directly) and return formatted contexts.
    """
    if not DDGS:
        return []

    try:
        results = []
        # Add retry logic
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    ddgs_gen = ddgs.text(query, max_results=REFERENCE_COUNT)
                    if ddgs_gen:
                        results = list(ddgs_gen)
                        if results:
                            break # Success
            except Exception as e:
                logger.warning(f"DuckDuckGo attempt {attempt+1} failed: {e}")

        logger.info(f"DuckDuckGo found {len(results)} results")

        if results:
             return [
                {
                    "id": str(uuid.uuid4()),
                    "name": r.get("title", "Source"),
                    "url": r.get("href", "#"),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]

        return []
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []

def get_related_questions(query: str, contexts: List[dict]) -> List[str]:
    """
    Generates related questions using the local LLM.
    """
    if not contexts:
        return []

    context_text = "\n\n".join([c["snippet"] for c in contexts])[:4000] # Limit context size

    prompt = ecEasyPrompts._more_questions_prompt.format(context=context_text)
    prompt += f"\n{query}"

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.7,
        )
        if not response.choices:
            return []
        content = response.choices[0].message.content

        # Log raw content for debugging
        logger.info(f"Related questions raw output: {content}")

        # Parse the output. We expect a list of questions.
        questions = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.endswith('?') or line.startswith('-') or line.startswith('*') or line[0].isdigit()):
                line = re.sub(r"^[*\-\d.]+\s*", "", line)
                questions.append(line)

        logger.info(f"Parsed {len(questions)} related questions")
        return questions[:3]
    except Exception as e:
        logger.warning(f"Related questions generation failed: {e}")
        return []

# ======== Generator Logic ========

def stream_response(
    query: str,
    search_uuid: str,
    generate_related_questions: bool
) -> Generator[str, None, None]:
    """
    Main logic to:
    1. Retrieve context (University RAG + Web fallback)
    2. Stream LLM Answer
    3. Stream Related Questions
    4. Cache results
    """

    # 1. Retrieve Contexts
    contexts = []

    # University-life RAG (priority)
    if vectorstore is not None:
        try:
            relevant_docs = vectorstore.similarity_search(query, k=4)
            university_contexts = [
                {
                    "id": str(uuid.uuid4()),
                    "name": "University Knowledge Base",
                    "url": "#",
                    "snippet": doc.page_content
                }
                for doc in relevant_docs
            ]
            contexts.extend(university_contexts)
            logger.info(f"University RAG found {len(university_contexts)} relevant chunks")
        except Exception as e:
            logger.error(f"University RAG error: {e}")

    # DuckDuckGo web search as fallback (only if not enough from RAG)
    if len(contexts) < REFERENCE_COUNT:
        try:
            web_results = search_with_duckduckgo(query)
            contexts.extend(web_results)
        except Exception as e:
            logger.error(f"Web search error: {e}")

    # Limit total contexts
    contexts = contexts[:REFERENCE_COUNT]

    # Send Contexts to client (for citations in UI)
    
    #yield json.dumps(contexts)
    #yield "\n\n__LLM_RESPONSE__\n\n"

    # 2. Prepare enhanced LLM Prompt
    context_block = "\n\n".join(
        [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
    )

    # Tailored system prompt for HK university life
    system_prompt = f"""You are a friendly, experienced student advisor for life in Hong Kong universities of Science and Technology (HKUST).
You know about dorms/halls, course registration, stress & mental health support, clubs & societies, part-time jobs, cost of living, transport (MTR/octopus), food on campus, social life, making friends, etc.

Use the following relevant information from university sources and web results if it helps answer accurately:
{context_block}

If the context doesn't fully cover the question, use your general knowledge but be honest if something might have changed recently (e.g. fees, rules in 2025–2026).

Answer in a supportive, encouraging, practical tone — like talking to a fellow student."""

    llm_response_accumulated = []

    try:
        client = get_llm_client()
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=1024,
            stop=STOP_WORDS,
            stream=True,
            temperature=0.7,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                llm_response_accumulated.append(content)
                yield content

    except Exception as e:
        logger.error(f"LLM Stream error: {e}")
        yield f"\n[Error generating response: {e}]"

    # 3. Related Questions
    related_questions_json = "[]"
    if SHOULD_DO_RELATED_QUESTIONS and generate_related_questions:
        try:
            questions = get_related_questions(query, contexts)
            formatted_questions = [{"question": q} for q in questions]
            related_questions_json = json.dumps(formatted_questions)
            #yield "\n\n__RELATED_QUESTIONS__\n\n"
            #yield related_questions_json
        except Exception as e:
            logger.error(f"Related questions error: {e}")

    # 4. Cache Result
    if search_uuid:
        full_response_data = [
            json.dumps(contexts),
            #"\n\n__LLM_RESPONSE__\n\n",
            "".join(llm_response_accumulated),
            #"\n\n__RELATED_QUESTIONS__\n\n" + related_questions_json
        ]
        try:
            with shelve.open(KV_NAME) as db:
                db[search_uuid] = full_response_data
        except Exception as e:
            logger.error(f"Cache write error: {e}")

# ======== FastAPI App ========

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    try:
        body = await request.json()
        logger.error(f"Request body: {body}")
    except:
        pass
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    # Check cache first
    if request.search_uuid:
        try:
            with shelve.open(KV_NAME) as db:
                if request.search_uuid in db:
                    cached_data = db[request.search_uuid]
                    return StreamingResponse(iter(cached_data), media_type="text/plain")
        except Exception:
            pass

    return StreamingResponse(
        stream_response(request.query, request.search_uuid, request.generate_related_questions),
        media_type="text/plain"
    )

@app.get("/")
def home():
    return RedirectResponse("/ui/index.html")

# Mount static files
if os.path.exists("ui"):
    app.mount("/ui", StaticFiles(directory="ui"), name="ui")
if os.path.exists("localData"):
    app.mount("/localData", StaticFiles(directory="localData"), name="localData")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)