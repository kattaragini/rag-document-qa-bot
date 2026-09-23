from pathlib import Path
import re
from pypdf import PdfReader
from .config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_documents():
    documents = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            documents.append({"source": path.name, "page": None, "text": clean_text(text)})
        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    documents.append({"source": path.name, "page": page_number, "text": text})
    return documents

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks

def build_chunks():
    output = []
    for document in load_documents():
        chunks = chunk_text(document["text"])
        for number, chunk in enumerate(chunks, start=1):
            output.append({
                "source": document["source"],
                "page": document["page"],
                "chunk_id": number,
                "text": chunk
            })
    return output
