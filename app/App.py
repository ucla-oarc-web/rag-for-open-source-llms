from fastapi import FastAPI, Body
from pydantic import BaseModel
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from custom_llm_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()


# Split text into smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=50,
    separator="\n\n"
)

doc_chunks = []
for file_path in Path("documents").glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")
    chunks = text_splitter.create_documents(
        [text],
        metadatas=[{"source": file_path.name}]
    )
    doc_chunks.extend(chunks)

# Create local embeddings & store in a local vector db. Knowledge is lost on quit
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(doc_chunks, embedding=embed_model)

# Create a custom prompt
# This is where you get to define what is sent to the llm. Remember, the llm/fm models do not access the knowledge base directly. You need to tell it what you want to know. This includes conversation. Unless you load the conversation as part of the query, the model will not have access to it.
CUSTOM_PROMPT = PromptTemplate(
    template="""
    INCLUDE
    You are an expert at answering questions about the Mythical Man Month.
    You will *only* answer questions using the given context.
    - If the user asks about something not in the context, respond exactly:
      I do not know.
    - Do not repeat the question in your answer.

    RESTRICT
    - Limit responses to 150 words or less.

    ADD
    Think about this step by step.

    REPEAT & POSITION
    Answer the following question using only the given context: {context}.
    Question: {question}
    Answer:
    """,
    input_variables=["context", "question"],
)

# Create a retrieval-based QA chain using Ollama as the LLM
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
ollama_llm = OllamaLLM(
    endpoint_url="http://127.0.0.1:11434",
    model="llama3.2",
    temperature=0.0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

# Pydantic model for incoming requests
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_llm(request: QueryRequest = Body(...)):
    """
    Ask a question and get an answer using the QA chain.
    """
    user_question = request.question
    result = qa_chain.invoke({"query": user_question})
    return result['result']
