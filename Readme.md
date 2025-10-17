# README

## Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) workflow using Ollama, FAISS, and LangChain. It provides a simple pipeline for vector-based semantic search and question-answering.

---

## Requirements
1. **Ollama**
   - Run `ollama serve` locally.
   - Confirm it’s running on localhost by visiting `http://127.0.0.1:11434/` in the browser; it should display **"Ollama is running"**.

2. **Docker**
   - Docker Compose configuration uses `network_mode: "host"`.
   - Ensure “Enable host networking” is enabled in your Docker settings.
   - Alternatively, configure port forwarding or SSH tunnels.

---

## Components
1. **FAISS**
   - Used as the in-memory vector store.
   - Stores and retrieves embeddings for semantic search.

2. **Embeddings**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face.

3. **Chunk Splitting**
   - Knowledge base documents are split into smaller chunks before embedding.

4. **Custom Prompting**
   - Demonstrates a custom prompt for LLM-driven Q&A.

5. **LangChain Example**
   - Shows how to integrate the above components in a LangChain pipeline.

---

## Usage

## Recreate Docker Instance
Use the following function to launch the Docker setup for the first time.

```bash
rag_build() {
  docker compose build
  docker compose -f docker-compose.yml up -d
}
```

## Recreate Docker Instance
Use the following function to tear down, rebuild, and relaunch the Docker setup.
* It would be useful to mount App.py, custom_llm_ollama.py, and requirements.txt so that you can update them live without requireing a rebuild.

```bash
rag_rebuild() {
  docker compose down
  docker compose build
  docker compose -f docker-compose.yml up -d
}
```

---

### RAG Request (Curl Wrapper)
```bash
rag_send() {
  if [ -z "$1" ]; then
    echo "Usage: rag_send \"<query>\""
    return 1
  fi

  local query="$1"
  curl -X POST -H "Content-Type: application/json" \
       -d "{\"question\": \"$query\"}" \
       http://localhost:8000/query
}
```
- **Example**:
  ```bash
    rag_send "Who is Anthony Doolan?"
    rag_send "How does adding more people to a software project impact the outcome?"
  ```
---

After rebuilding, confirm that your containers and Ollama are running correctly before sending queries.
