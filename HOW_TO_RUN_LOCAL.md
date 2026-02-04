# How to Run Netty Chat Locally (Without Lepton AI)

This setup allows you to run the Netty Chat backend locally using **Ollama**, **OpenAI**, or **DeepSeek** as the LLM provider.

## 1. Prerequisites

- Python 3.10+
- (Optional) Use a virtual environment:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate  # Windows
  # source .venv/bin/activate  # Linux/Mac
  ```

## 2. Configuration

Edit the `.env` file to select your LLM provider and configure keys.

### Example for OpenAI:
```dotenv
LLM_PROVIDER="openai"
OPENAI_API_KEY="sk-your-openai-api-key"
OPENAI_MODEL="gpt-4o"
```

### Example for DeepSeek:
```dotenv
LLM_PROVIDER="deepseek"
DEEPSEEK_API_KEY="sk-your-deepseek-api-key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"
```

### Example for Ollama (Local):
```dotenv
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_MODEL="qwen3:4b" 
# Make sure you have pulled the model in Ollama: `ollama pull qwen3:4b`
```

## 3. Installation & Run

We have provided a script `run_local_server.bat` for Windows users.

Double-click `run_local_server.bat` or run in terminal:

```bash
.\run_local_server.bat
```

Or manually:

```bash
pip install -r requirements_local.txt
python netty_local_server.py
```

The server will start at `http://0.0.0.0:8000`.

## 4. Accessing the UI

Open your browser and navigate to:
[http://localhost:8000/ui/index.html](http://localhost:8000/ui/index.html)

## Notes

- **Lepton AI Removal**: The `netty_local_server.py` file is a complete replacement for `chat_with_netty.py` and does not depend on `leptonai`.
- **RAG**: The Retrieval-Augmented Generation relies on the local ChromaDB database in `./arag/chromaVectorStore`. The server will attempt to load it. If it fails, RAG will be disabled but the chat will still work.

