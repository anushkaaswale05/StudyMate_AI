from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("🔎 Loading ChromaDB...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("✅ Retriever ready!")


if __name__ == "__main__":
    query = input("\nAsk something about your PDF: ")

    results = retriever.invoke(query)

    print(f"\n🔍 Found {len(results)} relevant chunks:\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:1000])
        print()