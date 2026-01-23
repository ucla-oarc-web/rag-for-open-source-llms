from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Split text into smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=50,
    separator="\n\n"
)

doc_chunks = []
for file_path in Path("../documents").glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")
    chunks = text_splitter.create_documents(
        [text],
        metadatas=[{"source": file_path.name}]
    )
    doc_chunks.extend(chunks)

# Create local embeddings & store in a local vector db. Knowledge is lost on quit
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(doc_chunks, embedding=embed_model)

# Shared retriever - this is the knowledge base
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
