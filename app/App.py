from fastapi import FastAPI, Body
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from custom_llm_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Create an in memory knowledge base
docs_text = """
Chocolate Chip Cookies: Preheat oven to 375°F (190°C), mix 2 1/4 cups all-purpose flour, 1 tsp baking soda, and 1/2 tsp salt; in another bowl, beat 1 cup softened butter, 3/4 cup granulated sugar, 3/4 cup brown sugar, and 1 tsp vanilla extract; add 2 large eggs one at a time; gradually mix in dry ingredients; stir in 2 cups chocolate chips; drop by spoonfuls onto ungreased baking sheets; bake for 8-10 minutes or until golden brown.
Brownies: Preheat oven to 350°F (175°C), melt 1/2 cup butter, stir in 1 cup sugar, 2 eggs, and 1 tsp vanilla extract; mix in 1/3 cup cocoa powder, 1/2 cup flour, 1/4 tsp salt, and 1/4 tsp baking powder; pour into a greased 8x8-inch pan; bake for 20-25 minutes or until the center is set. Enjoy!
"""

# Split text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
doc_chunks = text_splitter.create_documents([docs_text])

# Create local embeddings & store in a local vector db. Knowledge is lost on quit
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(doc_chunks, embedding=embed_model)

# Create a custom prompt
# This is where you get to define what is sent to the llm. Remember, the llm/fm models do not access the knowledge base directly. You need to tell it what you want to know. This includes conversation. Unless you load the conversation as part of the query, the model will not have access to it.
CUSTOM_PROMPT = PromptTemplate(
    template="""
    You are an expert at making desserts. You will only answer questions about making desserts.
    If the user asks you about something else, respond: "I am not able to answer that question."
    If you do not know the answer, say: "I do not know."

    Think about this step by step.

    Answer the following question using only the given context: {context}.
    Question: {question}
    Answer:
    """,
    input_variables=["context", "question"],
)

# Create a retrieval-based QA chain using Ollama as the LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
ollama_llm = OllamaLLM(
    endpoint_url="http://127.0.0.1:11434",
    model="llama3.2",
    temperature=0.7
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT},
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
