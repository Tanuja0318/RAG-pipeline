# ingest.py — Reads documents, splits them into chunks, embeds and stores in ChromaDB

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def load_documents():
    """Load all PDF and TXT files from the docs/ folder"""
    print(f"📂 Loading documents from: {DOCS_DIR}")

    # Load PDFs
    pdf_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    # Load TXT files
    txt_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    docs = []
    try:
        docs += pdf_loader.load()
    except Exception as e:
        print(f"PDF load warning: {e}")
    try:
        docs += txt_loader.load()
    except Exception as e:
        print(f"TXT load warning: {e}")

    print(f"✅ Loaded {len(docs)} document pages")
    return docs


def split_documents(docs):
    """Split documents into small chunks for better retrieval accuracy"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Each chunk = ~500 characters
        chunk_overlap=50,      # 50 chars overlap avoids cutting mid-sentence
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks):
    """Convert chunks to vectors using HuggingFace and store in ChromaDB"""

    # Free local embedding model — no API key needed for embeddings!
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",   # Small, fast, accurate
        model_kwargs={"device": "cpu"}
    )

    # Store in ChromaDB (persists to disk so you don't re-embed every time)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="rag_docs"
    )

    print(f"💾 Stored {len(chunks)} chunks in ChromaDB at: {CHROMA_DIR}")
    return vectorstore


def run_ingestion():
    docs = load_documents()
    if not docs:
        print("❌ No documents found. Add files to the docs/ folder.")
        return
    chunks = split_documents(docs)
    embed_and_store(chunks)
    print("🎉 Ingestion complete! Ready to query.")


if __name__ == "__main__":
    run_ingestion()