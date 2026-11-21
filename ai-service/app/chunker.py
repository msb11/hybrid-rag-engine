from io import BytesIO
from pypdf import PdfReader

def extract_and_chunk_pdf(file_bytes: bytes, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    reader = PdfReader(BytesIO(file_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    full_text = " ".join(full_text.split())  
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]