import streamlit as st
from src.config import INDEX_FILE, METADATA_FILE
from src.rag import ask

st.set_page_config(page_title="Basic Document Q&A Bot", page_icon="📚")
st.title("📚 Basic Document Q&A Bot")
st.caption("RAG pipeline with document retrieval, grounded generation, and source citations.")

if not INDEX_FILE.exists() or not METADATA_FILE.exists():
    st.warning("Index not found. Run `python -m scripts.build_index` first.")
    st.stop()

question = st.text_input("Ask a question about the documents")
if question:
    with st.spinner("Retrieving and generating..."):
        answer, results = ask(question)
    st.markdown(answer)
    st.subheader("Retrieved source chunks")
    for item in results:
        location = f"Page {item['page']}" if item["page"] else f"Chunk {item['chunk_id']}"
        with st.expander(f"{item['source']} — {location} — score {item['score']:.3f}"):
            st.write(item["text"])
