import os
import json
import time
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from app.db import get_conn

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_query(query: str) -> np.ndarray:
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=768,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    vec = np.array(res.embeddings[0].values, dtype=np.float32)
    return vec / np.linalg.norm(vec)

def check_cache(query_emb: np.ndarray, threshold: float = 0.92):
    sql = """
    SELECT answer, sources, 1 - (query_embedding <=> %s) AS similarity
    FROM query_cache
    WHERE 1 - (query_embedding <=> %s) >= %s
    ORDER BY similarity DESC
    LIMIT 1;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_emb, query_emb, threshold))
            row = cur.fetchone()
            if row:
                return {"answer": row[0], "sources": row[1], "similarity": round(float(row[2]), 4)}
    return None

def save_cache(query_text: str, query_emb: np.ndarray, answer: str, sources: list[dict]):
    sql = """
    INSERT INTO query_cache (query_text, query_embedding, answer, sources)
    VALUES (%s, %s, %s, %s::jsonb);
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_text, query_emb, answer, json.dumps(sources)))
        conn.commit()

def retrieve_chunks(query_text: str, query_emb: np.ndarray, mode: str = "hybrid_rrf", top_k: int = 5):
    t0 = time.perf_counter()
    with get_conn() as conn:
        with conn.cursor() as cur:
            if mode == "vector_only":
                cur.execute(
                    """
                    SELECT c.id, d.title, c.content, 1 - (c.embedding <=> %s) AS score
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY c.embedding <=> %s
                    LIMIT %s;
                    """,
                    (query_emb, query_emb, top_k),
                )
            elif mode == "fts_only":
                cur.execute(
                    """
                    SELECT c.id, d.title, c.content, ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) AS score
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.tsv @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s;
                    """,
                    (query_text, query_text, top_k),
                )
            else:  # hybrid_rrf (Reciprocal Rank Fusion, k=60)
                cur.execute(
                    """
                    WITH vector_search AS (
                        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS rank
                        FROM chunks
                        ORDER BY embedding <=> %s
                        LIMIT 30
                    ),
                    fts_search AS (
                        SELECT id, ROW_NUMBER() OVER (
                            ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) DESC
                        ) AS rank
                        FROM chunks
                        WHERE tsv @@ websearch_to_tsquery('english', %s)
                        LIMIT 30
                    ),
                    rrf AS (
                        SELECT
                            COALESCE(v.id, f.id) AS id,
                            COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + f.rank), 0.0) AS score
                        FROM vector_search v
                        FULL OUTER JOIN fts_search f ON v.id = f.id
                    )
                    SELECT c.id, d.title, c.content, r.score
                    FROM rrf r
                    JOIN chunks c ON c.id = r.id
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY r.score DESC
                    LIMIT %s;
                    """,
                    (query_emb, query_emb, query_text, query_text, top_k),
                )
            rows = cur.fetchall()

    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
    sources = [
        {"chunk_id": r[0], "document": r[1], "content": r[2], "score": round(float(r[3]), 4)}
        for r in rows
    ]
    return sources, retrieval_ms

def generate_answer(query_text: str, sources: list[dict]) -> str:
    if not sources:
        return "No relevant context found in the indexed documents."

    context = "\n\n".join(
        f"[Chunk {s['chunk_id']} | {s['document']}]\n{s['content']}" for s in sources
    )
    prompt = f"""Answer the user's question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information in the indexed documents."
Cite chunk IDs like [Chunk 12] when stating facts.

Context:
{context}

Question: {query_text}"""

    res = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )
    return res.text