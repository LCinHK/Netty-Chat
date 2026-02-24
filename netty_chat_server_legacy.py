"""
This module implements the core backend server for ECEasy, a Retrieval-Augmented
Generation (RAG) application. It uses FastAPI to create a web server that provides
endpoints for querying a Large Language Model (LLM) with context retrieved from various
sources.

Key Features:
- FastAPI-based web server.
- A RAG pipeline that combines context from a vector store and web search.
- Caching mechanism using a file-based key-value store (shelve).
- Streaming API responses for real-time interaction.
- Generation of related questions based on the query and context.
- Serves a static frontend for the chat interface.
"""

import concurrent.futures
import json
import os
import re
import threading
import traceback
from typing import Annotated, List, Generator, Optional
import uuid

# ======== FastAPI Imports ========
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import httpx
from loguru import logger

# ======== KV Imports ========
import shelve

# ======== Search Engine Functions ========
from langchain_community.tools import DuckDuckGoSearchResults

# ======== Prompt Texts ========
import nettyPrompts
from arag.arag import get_rag_context

# ======== Environment and Constants ========
from dotenv import load_dotenv
load_dotenv(override=True)
logger.info("Loaded .env file successfully.")

# Number of references to fetch from the search engine.
REFERENCE_COUNT = 8
# A set of stop words for the LLM generation.
STOP_WORDS = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]
# Key-value store file name for caching.
KV_NAME = os.environ.get("KV_NAME", "netty-chat.kv")
# LLM model configuration.
LLM_MODEL = os.environ.get("LLM_MODEL", "mixtral-8x7b")
# Flag to determine if related questions should be generated.
SHOULD_DO_RELATED_QUESTIONS = os.environ.get("RELATED_QUESTIONS", "true").lower() == "true"


# ======== Search Functions ========

def search_with_duckduckgo(query: str):
    """
    Performs a web search using DuckDuckGo and formats the results.
    """
    search = DuckDuckGoSearchResults(output_format="list", num_results=REFERENCE_COUNT)
    results = search.invoke(query)
    return [{"name": r["title"], "url": r["link"], "snippet": r["snippet"]} for r in results]

def search_with_adaptiveRAG(query: str):
    """
    Retrieves context from the local vector store.
    """
    return get_rag_context(query)


# ======== OpenAI Client Management ========

def get_openai_client(force_openai=False):
    """
    Gets a thread-local OpenAI client. This ensures that each thread has its
    own client, which is important for thread safety.
    """
    import openai
    thread_local = threading.local()
    if hasattr(thread_local, 'client') and not force_openai:
        return thread_local.client

    if os.environ.get("LLM_USE_CUSTOM_SERVER"):
        logger.info(f"Using custom LLM model. Remote URL: {os.environ['LLM_REMOTE_URL']}")
        client = openai.OpenAI(
            base_url=os.environ["LLM_REMOTE_URL"],
            api_key=os.environ.get("LLM_REMOTE_API_KEY"),
            timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
        )
    else:
        logger.info("Using default LLM provider.")
        # This part might need adjustment based on the desired default provider
        # For now, it's set up for a generic OpenAI-compatible API.
        client = openai.OpenAI(
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
        )

    thread_local.client = client
    return client


# ======== Core RAG Logic ========

def get_related_questions(query: str, contexts: List[dict]) -> List[str]:
    """
    Generates related questions based on the original query and retrieved contexts
    using an LLM call.
    """
    # This function uses a placeholder for the tool definition,
    # as the original `leptonai.util.tool` is not available.
    # A direct JSON output instruction is a common alternative.
    system_prompt = (
        "You are a helpful assistant. Based on the provided context and user query, "
        "generate a list of 3 relevant follow-up questions that a user might ask. "
        "Return the questions as a JSON object with a key 'questions' containing a list of strings. "
        "For example: {\"questions\": [\"What is the first question?\", \"What is the second question?\"]}"
        f"Context:\n\n" + "\n\n".join([c["snippet"] for c in contexts])
    )

    try:
        client = get_openai_client(force_openai=True) # Use a client that can handle this
        response = client.chat.completions.create(
            model=LLM_MODEL, # Use the model from .env
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=512,
            temperature=0.5,
        )
        content = response.choices[0].message.content
        if content:
            related = json.loads(content)
            logger.trace(f"Related questions: {related}")
            return related.get("questions", [])
        return []
    except Exception as e:
        logger.error(f"Error generating related questions: {e}\n{traceback.format_exc()}")
        return []


def stream_and_cache_response(
    contexts: List[dict],
    llm_response: Generator,
    related_questions_future: Optional[concurrent.futures.Future],
    search_uuid: str
) -> Generator[str, None, None]:
    """
    Streams the RAG response (contexts, LLM answer, related questions) to the client
    while concurrently caching the full response in a key-value store.
    """
    # 1. Yield contexts
    yield json.dumps(contexts)
    yield "\n\n__LLM_RESPONSE__\n\n"

    # 2. Stream LLM response
    llm_output = []
    if not contexts:
        warning = "(The search engine returned nothing for this query. Please take the answer with a grain of salt.)\n\n"
        llm_output.append(warning)
        yield warning

    for chunk in llm_response:
        content = chunk.choices[0].delta.content or ""
        llm_output.append(content)
        yield content

    # 3. Yield related questions
    related_questions_output = ""
    if related_questions_future:
        related_questions = related_questions_future.result()
        try:
            result_json = json.dumps(related_questions)
            related_questions_output = "\n\n__RELATED_QUESTIONS__\n\n" + result_json
            yield related_questions_output
        except Exception as e:
            logger.error(f"Error serializing related questions: {e}\n{traceback.format_exc()}")

    # 4. Cache the complete response
    full_response = [
        json.dumps(contexts),
        "\n\n__LLM_RESPONSE__\n\n",
        "".join(llm_output),
        related_questions_output
    ]
    try:
        with shelve.open(KV_NAME) as db:
            db[search_uuid] = full_response
        logger.info(f"Successfully cached response for search_uuid: {search_uuid}")
    except Exception as e:
        logger.error(f"Failed to cache response for {search_uuid}: {e}")


# ======== FastAPI Application Setup ========

app = FastAPI(
    title="ECEasy Server",
    description="A standalone FastAPI server for the ECEasy RAG application.",
)

# Thread pool for background tasks like generating related questions.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

# Initialize KV store on startup
try:
    with shelve.open(KV_NAME) as db:
        logger.info(f"KV store '{KV_NAME}' created/loaded. Current number of keys: {len(db)}")
except Exception as e:
    logger.error(f"Could not initialize KV store '{KV_NAME}': {e}")


@app.post("/query")
def query_handler(
    query: str,
    search_uuid: str,
    generate_related_questions: Optional[bool] = True,
):
    """
    Handles the main RAG query. It orchestrates context retrieval, LLM generation,
    and response streaming.
    """
    # 1. Check cache first
    if search_uuid:
        try:
            with shelve.open(KV_NAME) as db:
                if search_uuid in db:
                    logger.info(f"Cache hit for search_uuid: {search_uuid}")
                    return StreamingResponse(content=iter(db[search_uuid]), media_type="text/html")
        except Exception as e:
            logger.error(f"Cache lookup failed, generating new response: {e}")
    else:
        raise HTTPException(status_code=400, detail="search_uuid must be provided.")

    # 2. Retrieve context
    # Sanitize query to prevent prompt injection attacks
    query = re.sub(r"\[/?INST\]", "", query) if query else nettyPrompts._default_query

    contexts = []
    try:
        # Combine local RAG with web search
        rag_contexts = search_with_adaptiveRAG(query)
        contexts.extend(rag_contexts)
        logger.info(f"Got {len(rag_contexts)} contexts from vector store.")

        if len(contexts) < REFERENCE_COUNT:
            web_contexts = search_with_duckduckgo(query)
            needed = REFERENCE_COUNT - len(contexts)
            contexts.extend(web_contexts[:needed])
            logger.info(f"Added {min(needed, len(web_contexts))} contexts from web search.")

    except Exception as e:
        logger.error(f"Error during context retrieval: {e}\n{traceback.format_exc()}")

    logger.debug(f"Final contexts: \n{json.dumps(contexts, indent=2)}")

    # 3. Prepare for LLM call
    system_prompt = nettyPrompts._rag_query_text.format(
        context="\n\n".join(
            [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
        )
    )

    try:
        client = get_openai_client()
        llm_response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=1024,
            stop=STOP_WORDS,
            stream=True,
            temperature=0.9,
        )

        # Start generating related questions in the background
        related_questions_future = None
        if SHOULD_DO_RELATED_QUESTIONS and generate_related_questions:
            related_questions_future = executor.submit(get_related_questions, query, contexts)

    except Exception as e:
        logger.error(f"Error calling LLM: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=503, detail="Error communicating with the Language Model.")

    # 4. Stream response
    return StreamingResponse(
        stream_and_cache_response(
            contexts, llm_response, related_questions_future, search_uuid
        ),
        media_type="text/html",
    )

# Serve static files for the UI
app.mount("/ui", StaticFiles(directory="ui"), name="ui")
app.mount("/localData", StaticFiles(directory="localData"), name="localData")

@app.get("/")
def index():
    """
    Redirects the root URL to the main chat interface.
    """
    return RedirectResponse(url="/ui/index.html")

# To run this server, use the command:
# uvicorn netty_chat_server:app --host 0.0.0.0 --port 8080

