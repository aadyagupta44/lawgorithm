from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import tempfile
import uuid

from ingestion import DocumentLoader, DocumentChunker, PineconeEmbedder
from graph import run_graph

app = FastAPI(
    title="Lawgorithm API",
    description="API for Intelligent Legal Document Assistant. Provides document ingestion and LangGraph augmented querying.",
    version="1.0.0",
)

# ---- Models ----
class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's legal question to ask.")
    namespace: str = Field(..., description="The Pinecone namespace generated from ingested documents.")
    confidence_threshold: Optional[float] = Field(0.7, description="Threshold to force retry loops if confidence is low.")
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="List of previous question and answer dicts.")

class QueryResponse(BaseModel):
    generation: str
    source_files: List[str]
    source_pages: List[int]
    loop_count: int
    confidence_score: Optional[float]
    relevance_score: Optional[float]
    hallucination_score: Optional[float]

class IngestTextRequest(BaseModel):
    text: str = Field(..., description="The raw legal text to ingest.")
    document_name: Optional[str] = Field("Pasted_Document", description="Name of the document.")

class IngestResponse(BaseModel):
    message: str
    namespace: str
    chunks_created: int
    vectors_stored: int

# ---- Endpoints ----

@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Uploads a PDF file, chunks it, and stores embeddings in Pinecone."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        docs = loader.load_pdf(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to load PDF: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(docs)
    
    embedder = PineconeEmbedder()
    namespace = embedder.namespace_from_filename(file.filename)
    stored_count = embedder.embed_and_store(chunks, namespace)
    
    return IngestResponse(
        message="File ingested successfully.",
        namespace=namespace,
        chunks_created=len(chunks),
        vectors_stored=stored_count
    )

@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestTextRequest):
    """Chunks and stores raw pasted text in Pinecone."""
    loader = DocumentLoader()
    docs = loader.load_text(request.text, request.document_name)
    
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(docs)
    
    embedder = PineconeEmbedder()
    namespace = embedder.namespace_from_filename(request.document_name)
    stored_count = embedder.embed_and_store(chunks, namespace)
    
    return IngestResponse(
        message="Text ingested successfully.",
        namespace=namespace,
        chunks_created=len(chunks),
        vectors_stored=stored_count
    )


@app.post("/query", response_model=QueryResponse, tags=["Querying"])
async def query_documents(request: QueryRequest):
    """Query ingested documents using LangGraph self-correcting RAG pipeline."""
    try:
        result = run_graph(
            question=request.question,
            namespace=request.namespace,
            chat_history=request.chat_history,
            confidence_threshold=request.confidence_threshold,
        )
        
        return QueryResponse(
            generation=result.get("generation", ""),
            source_files=result.get("source_files", []),
            source_pages=result.get("source_pages", []),
            loop_count=result.get("loop_count", 0),
            confidence_score=result.get("confidence_score"),
            relevance_score=result.get("relevance_score"),
            hallucination_score=result.get("hallucination_score")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing graph: {str(e)}")
