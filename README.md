# Basic Document Q&A Bot — RAG Pipeline

## 1. What this project does

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline for question answering over a local document collection. It loads PDF/TXT documents, cleans and chunks the text, creates embeddings in batches, stores them in a persistent FAISS vector index, retrieves the most relevant chunks for a user question, and generates a grounded answer using Google Gemini. Answers include source filenames and page/chunk locations.

## 2. Tech stack

- Python 3.11+
- `pypdf` — PDF text extraction
- `sentence-transformers` — local text embeddings
- `faiss-cpu` — persistent vector similarity search
- `google-genai` — Gemini answer generation
- `python-dotenv` — environment configuration
- Optional `streamlit` can be installed for the web UI

## 3. Architecture

```text
Documents (PDF/TXT)
        |
        v
Document ingestion + text cleaning
        |
        v
Chunking with overlap + metadata
        |
        v
Batch embeddings (Sentence Transformers)
        |
        v
Persistent FAISS vector index
        |
        | user question
        v
Question embedding
        |
        v
Top-k similarity retrieval
        |
        v
Retrieved chunks + source metadata
        |
        v
Gemini grounded generation
        |
        v
Answer + citations
```

## 4. Chunking strategy

This implementation uses fixed-size word chunks with overlap. The default is 900 words with 150 words of overlap. Fixed-size chunking is simple and predictable for mixed TXT/PDF documents, while overlap reduces the chance of losing context at chunk boundaries. The values can be changed with `CHUNK_SIZE` and `CHUNK_OVERLAP`.

## 5. Embedding model and vector database

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. All document chunks are encoded together using a batch size of 32, satisfying the batching requirement and avoiding one embedding call per chunk. FAISS `IndexFlatIP` is used with normalized vectors, so inner product corresponds to cosine similarity. The FAISS index and metadata are persisted under `storage/`.

## 6. Setup — local/GitHub clone

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd basic-document-qa-rag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your own Gemini API key:

```text
GEMINI_API_KEY=your_key_here
```

Never commit `.env` or any API key.

## 7. Build the index

```bash
python -m scripts.build_index
```

This is the indexing/ingestion step. It loads the documents, chunks them, embeds all chunks in batches, and saves the FAISS index plus metadata.

## 8. Run the command-line bot

```bash
python -m scripts.chat
```

Try these questions:
1. What is the difference between a list and a tuple in Python?
2. How can overfitting be reduced in machine learning?
3. What is the purpose of a LEFT JOIN?
4. What is the shared responsibility model in cloud computing?
5. Why does a RAG system use chunk overlap?
6. What are the ACID properties?
7. Which document explains source citations in RAG?
8. What is the capital of France? (This should be refused because it is outside the supplied knowledge base.)

## 9. Optional Streamlit UI

Install Streamlit:

```bash
pip install streamlit
```

Then:

```bash
streamlit run app.py
```

The UI displays the answer and the retrieved source chunks.

## 10. Google Colab

Open `notebooks/rag_colab.ipynb` in Google Colab. Upload/clone this repository, install dependencies, set `GEMINI_API_KEY` in the Colab environment, build the index, and run the same Q&A flow.

For a GitHub-hosted notebook, use the GitHub repository URL and open the notebook with Colab.

## 11. Environment variables

- `GEMINI_API_KEY` — required for answer generation
- `GEMINI_MODEL` — default `gemini-2.5-flash`
- `EMBEDDING_MODEL` — default `sentence-transformers/all-MiniLM-L6-v2`
- `TOP_K` — number of retrieved chunks, default 4
- `CHUNK_SIZE` — words per chunk, default 900
- `CHUNK_OVERLAP` — overlapping words, default 150

## 12. Known limitations

- PDF extraction may be imperfect for scanned/image-only PDFs.
- Retrieval quality depends on the embedding model and chunking parameters.
- The demo uses a single FAISS index and is intended for an internship assignment, not a production multi-user deployment.
- Gemini generation requires an API key and internet access.
- The bot is intentionally instructed not to answer from outside knowledge when the supplied context does not contain the answer.

## 13. Deliverables checklist

- Public GitHub repository
- `README.md`
- 5 meaningful documents under `data/docs/`
- At least one PDF
- Source code under `src/` and `scripts/`
- Google Colab notebook under `notebooks/`
- Screen recording showing ingestion/indexing, startup, at least five questions across two or more documents, citations, an unanswerable question, and one technical design decision
- Reply to the hiring email with the GitHub URL and accessible screen-recording URL

## 14. Important integrity note

The assignment explicitly permits AI coding assistants, but asks candidates to understand and explain their implementation. Before recording the demo, run the project yourself and make sure you can explain ingestion, chunking, embeddings, FAISS retrieval, top-k, and grounded generation.
