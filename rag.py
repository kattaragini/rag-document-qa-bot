import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from .config import (
    STORAGE_DIR, INDEX_FILE, METADATA_FILE, EMBEDDING_MODEL,
    GEMINI_MODEL, TOP_K
)
from .ingest import build_chunks

def build_index():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    texts = [item["text"] for item in chunks]

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vectors = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    METADATA_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len({c['source'] for c in chunks})} documents.")
    print(f"Saved: {INDEX_FILE}")

def retrieve(question, top_k=TOP_K):
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise RuntimeError("Index not found. Run: python -m scripts.build_index")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = embedder.encode([question], normalize_embeddings=True)
    index = faiss.read_index(str(INDEX_FILE))
    scores, indices = index.search(query_vector, top_k)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = metadata[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)
    return results

def generate_answer(question, results):
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

    context_parts = []
    for i, item in enumerate(results, start=1):
        location = f"page {item['page']}" if item["page"] else f"chunk {item['chunk_id']}"
        context_parts.append(
            f"[SOURCE {i}] {item['source']} ({location})\n{item['text']}"
        )
    context = "\n\n".join(context_parts)

    client = genai.Client(api_key=api_key)
    prompt = f"""You are a grounded document Q&A assistant.

Answer the user's question using ONLY the supplied document context.
If the context does not contain enough information, say:
"I could not find that information in the provided documents."
Do not use outside knowledge to fill missing information.

Question:
{question}

Document context:
{context}

Give a concise, clear answer. At the end, add a Sources section listing the source filenames and page/chunk locations you actually used.
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text

def ask(question):
    results = retrieve(question)
    answer = generate_answer(question, results)
    return answer, results
