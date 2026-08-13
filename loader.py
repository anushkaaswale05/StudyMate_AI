from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


DOCUMENTS_DIR = Path("documents")


def load_documents():
    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the documents folder.")
        return documents

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        documents.extend(pages)

    print(f"\nLoaded {len(documents)} pages from {len(pdf_files)} PDF(s).")

    return documents


if __name__ == "__main__":
    load_documents()