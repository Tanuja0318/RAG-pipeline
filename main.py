# main.py — FastAPI server with endpoints for chat and document upload

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Allow frontend (HTML file) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, replace with your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend HTML file
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Request/response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/")
def root():
    """Serve the chat UI"""
    return FileResponse("../frontend/index.html")


@app.get("/health")
def health():
    """Health check — useful for Docker and monitoring"""
    return {"status": "ok", "message": "RAG Chatbot is running!"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint: takes a question, runs RAG pipeline, returns answer + sources
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Import here to avoid loading model at startup if ChromaDB is empty
        from rag_pipeline import query_rag
        result = query_rag(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or TXT), save it, then re-run ingestion
    """
    # Only allow PDF and TXT
    allowed = [".pdf", ".txt"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files allowed")

    # Save file to docs/ folder
    docs_dir = os.getenv("DOCS_DIR", "./docs")
    os.makedirs(docs_dir, exist_ok=True)
    file_path = os.path.join(docs_dir, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Re-run ingestion to include new document
    from ingest import run_ingestion
    run_ingestion()

    return {"message": f"✅ '{file.filename}' uploaded and indexed successfully!"}


@app.get("/documents")
def list_documents():
    """List all documents currently loaded in the system"""
    docs_dir = os.getenv("DOCS_DIR", "./docs")
    if not os.path.exists(docs_dir):
        return {"documents": []}
    files = [f for f in os.listdir(docs_dir) if f.endswith((".pdf", ".txt"))]
    return {"documents": files}