from splitter import split_documents
from langchain_huggingface import HuggingFaceEmbeddings

print("🔢 Creating embeddings...")

# Get our 137 text chunks
chunks = split_documents()

if not chunks:
    print("❌ No chunks found.")
    exit()

print(f"📄 Chunks received: {len(chunks)}")

# Load a lightweight local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("🧠 Embedding model loaded.")

# Convert the first chunk to a vector as a test
vector = embeddings.embed_query(chunks[0].page_content)

print(f"✅ Embedding created!")
print(f"Vector dimensions: {len(vector)}")
print("🎉 Embedding step completed!")