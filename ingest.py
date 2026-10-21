import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    DOCUMENTS_FOLDER, INDEX_FOLDER,
)

# The "meaning to numbers" tool
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# The scissors that cut text into pieces
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_all_pdfs(folder: str):
    """Step 1: read every PDF in the folder. Returns a list of pages."""
    pages = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder, filename)
            loaded = PyPDFLoader(path).load()          # one item per page
            for page in loaded:
                page.metadata["source"] = filename      # remember which PDF
            pages.extend(loaded)
            print(f"Read {filename}: {len(loaded)} pages")
    return pages


def build_index():
    """Steps 2-4: cut into chunks, convert to numbers, save in FAISS."""
    pages = load_all_pdfs(DOCUMENTS_FOLDER)
    if not pages:
        raise RuntimeError(f"No PDFs found in '{DOCUMENTS_FOLDER}'. Add some first.")

    chunks = splitter.split_documents(pages)            # 2. CUT
    print(f"Created {len(chunks)} chunks")

    store = FAISS.from_documents(chunks, embeddings)    # 3. NUMBERS  + 4. STORE

    if os.path.exists(INDEX_FOLDER):                    # start clean
        shutil.rmtree(INDEX_FOLDER)
    store.save_local(INDEX_FOLDER)
    print(f"Saved index to '{INDEX_FOLDER}'")


def load_index():
    """Open the saved cabinet so we can search it."""
    return FAISS.load_local(
        INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    build_index()