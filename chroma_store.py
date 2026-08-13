from splitter import split_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("🚀 Starting ChromaDB setup...")

# Load document chunks
chunks = split_documents()

if not chunks:
    print("❌ No chunks found.")
    exit()

print(f"📄 Chunks received: {len(chunks)}")

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("🧠 Embedding model loaded.")

# Create persistent Chroma database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ ChromaDB created successfully!")
print(f"📦 Stored {len(chunks)} document chunks.")
print("🎉 Vector database is ready!")