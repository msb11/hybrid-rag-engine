import os
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.chunker import extract_and_chunk_pdf
from app.db import save_document_and_chunks
from google.genai.errors import APIError

from pydantic import BaseModel
from typing import Literal
from app.retriever import embed_query, check_cache, save_cache, retrieve_chunks, generate_answer

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI(title="Hybrid RAG Engine - AI Service")

def embed_texts(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        while True:
            try:
                res = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=768,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                break
            except APIError as e:
                if e.code == 429:
                    print("Rate limit hit (100/min). Waiting 60s before next batch...")
                    time.sleep(60)
                else:
                    raise

        for emb in res.embeddings:
            vec = np.array(emb.values, dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            all_embeddings.append(vec.tolist())
    return all_embeddings

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    t0 = time.perf_counter()
    file_bytes = await file.read()
    chunks = extract_and_chunk_pdf(file_bytes)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    embeddings = embed_texts(chunks)
    doc_id = save_document_and_chunks(file.filename, chunks, embeddings)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "document_id": doc_id,
        "title": file.filename,
        "chunk_count": len(chunks),
        "elapsed_ms": elapsed_ms,
    }

class SearchRequest(BaseModel):
    query: str
    mode: Literal["hybrid_rrf", "vector_only", "fts_only"] = "hybrid_rrf"
    top_k: int = 5
    use_cache: bool = True

@app.post("/search")
async def search_docs(req: SearchRequest):
    t0 = time.perf_counter()
    query_emb = embed_query(req.query)

    if req.use_cache:
        cached = check_cache(query_emb)
        if cached:
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "mode": req.mode,
                "cache_hit": True,
                "retrieval_ms": round((time.perf_counter() - t0) * 1000, 2),
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    sources, retrieval_ms = retrieve_chunks(req.query, query_emb, mode=req.mode, top_k=req.top_k)
    answer = generate_answer(req.query, sources)

    if req.use_cache and sources:
        save_cache(req.query, query_emb, answer, sources)

    return {
        "answer": answer,
        "sources": sources,
        "mode": req.mode,
        "cache_hit": False,
        "retrieval_ms": retrieval_ms,
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }