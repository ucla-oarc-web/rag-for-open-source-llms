# README

## Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) workflow using Ollama, FAISS, and LangChain. It provides a simple pipeline for vector-based semantic search and question-answering.

To create the documents in ./app/documents I used (The Mythical Manmonth PDF)[https://ia801600.us.archive.org/12/items/MythicalManMonth/Brooks%201974%20The%20Mythical%20Man-Month-%20Essays%20on%20Software%20Engineering%2C%20Anniversary%20Edition%20%282nd%20Edition%29.pdf] from https://us.archive.org.

I manually copied each chapter into its own tet file to help the retravial process get context based on chapters. It wpuld be a simple change to update the knowledge base to use a PDF loader from pypdf to load the PDF automatically into text, then use CharacterTextSplitter to load the text into the vectore store. Even better if you use langchains RecursiveCharacterTextSplitter.

---

## Requirements
1. **Ollama**
   - Run `ollama serve` locally.
   - Confirm it’s running on localhost by visiting `http://127.0.0.1:11434/` in the browser; it should display **"Ollama is running"**.

2. **Docker**
   - Docker Compose configuration uses `network_mode: "host"`.
   - Ensure “Enable host networking” is enabled in your Docker settings.
   - Alternatively, configure port forwarding or SSH tunnels.

3. **Documents for the knowledge base**
   - You will get an error if you do not create the document folder and add txt documents at `app/documents`.
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
  curl -s -X POST -H "Content-Type: application/json" \
       -d "{\"question\": \"$query\"}" \
       http://localhost:8000/query | jq -r '.'
}
```
- **Example**:
  ```bash
    rag_send "Who is Anthony Doolan?"
    rag_send "How does adding more people to a software project impact the outcome?"
  ```

### Agent Request (Curl Wrapper)
```bash
agent_send() {
  if [ -z "$1" ]; then
    echo "Usage: agent_send \"<query>\" [session_id]"
    return 1
  fi

  local query="$1"
  local session_id="${2:-chat1}"

  curl -s -X POST -H "Content-Type: application/json" \
       -d "{\"question\": \"$query\", \"session_id\": \"$session_id\"}" \
       http://localhost:8000/agent | jq -r '"Tools: \(.tools_used | join(", "))\nSession: \(.session_id // "none")\n\nAnswer: \(.answer)"'
}
```
- **Example**:
  ```bash
    # Uses default session "chat1"
    agent_send "Who is Anthony Doolan at OARC UCLA?"
    agent_send "What is Brooks' Law?"
    agent_send "Can you explain that further?"

    # Start a new conversation with a different session
    agent_send "What is Brooks' Law from the Mythical Man Month, and are there any recent news articles about companies experiencing this problem?" chat2
  ```

- **Debugging the Agent**:
  ```bash
  docker logs langchain_ollama
  ```

  Message Types show:
   * Type: human = the original question from the human.
   * Type: ai = Tool requests and thoughts.
   * Type: tool = Tool function output send to the agent.
---

After rebuilding, confirm that your containers and Ollama are running correctly before sending queries.
