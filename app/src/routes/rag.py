from fastapi import APIRouter, Body
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from shared.knowledge_base import retriever

router = APIRouter()

# RAG-specific prompt
RAG_PROMPT = PromptTemplate(
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

ollama_llm = OllamaLLM(
    base_url="http://127.0.0.1:11434",
    model="llama3.1:8b",
    temperature=0.0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain with its own prompt
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | ollama_llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query_llm(request: QueryRequest = Body(...)):
    """
    Ask a question and get an answer using the QA chain.
    """
    result = qa_chain.invoke(request.question)
    return result
