from fastapi import FastAPI
from routes.rag import router as rag_router
from routes.agent import router as agent_router

app = FastAPI()

# Simple RAG pattern using FAISS Knowledge base and LLM
app.include_router(rag_router)

# Simple Langchain Agent Pattern using FAISS Knowledge base, DuckDuckGo, and LLM
app.include_router(agent_router)
