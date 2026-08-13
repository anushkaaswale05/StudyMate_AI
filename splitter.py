from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("documents/AI_Notes.pdf")


def split_documents():
    print("📖 Loading PDF...")

    if not PDF_PATH.exists():
        print("❌ PDF not found!")
        return []

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} pages")

    print("✂️ Splitting pages into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")

    return chunks


if __name__ == "__main__":
    chunks = split_documents()

    if chunks:
        print("\n📌 First chunk preview:")
        print(chunks[0].page_content[:500])

    print("\n🎉 Splitting completed successfully!")