import os
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def save_document_and_chunks(title: str, chunks: list[str], embeddings: list[list[float]]) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (title, chunk_count) VALUES (%s, %s) RETURNING id;",
                (title, len(chunks)),
            )
            doc_id = str(cur.fetchone()[0])

            rows = [(doc_id, text, emb) for text, emb in zip(chunks, embeddings)]
            cur.executemany(
                "INSERT INTO chunks (document_id, content, embedding) VALUES (%s, %s, %s);",
                rows,
            )
        conn.commit()
    return doc_id