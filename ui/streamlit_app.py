"""Step 5b: the chat page.

Start the API first (python -m uvicorn app.main:app), then in a second window:
    python -m streamlit run ui/streamlit_app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from ui.api_client import ApiClient, ApiError

st.set_page_config(page_title="RAG Document Q&A", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history
if "uploaded_ids" not in st.session_state:
    st.session_state.uploaded_ids = set()  # files already sent to the API


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}**, page {s['page']} (similarity {s['score']})")
            st.caption(s["snippet"])


# ---------- sidebar: connection, upload, document list ----------
with st.sidebar:
    st.header("Documents")
    api_url = st.text_input("API address", os.getenv("API_URL", "http://127.0.0.1:8000"))
    client = ApiClient(api_url)

    try:
        stats = client.health()
        st.success(f"API online: {stats['documents']} document(s), {stats['chunks']} chunks")
    except ApiError as err:
        st.error(str(err))
        st.stop()

    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    for file in uploads or []:
        if file.file_id in st.session_state.uploaded_ids:
            continue
        with st.spinner(f"Reading {file.name}..."):
            try:
                info = client.upload(file.name, file.getvalue())
                st.session_state.uploaded_ids.add(file.file_id)
                st.toast(f"{info['filename']}: {info['pages']} page(s), {info['chunks']} chunks")
            except ApiError as err:
                st.error(f"{file.name}: {err}")

    documents = client.list_documents()
    for doc in documents:
        name_col, button_col = st.columns([4, 1])
        name_col.write(f"{doc['filename']}  \n{doc['chunks']} chunks")
        if button_col.button("✕", key=f"delete-{doc['filename']}", help="Remove this document"):
            try:
                client.delete(doc["filename"])
            except ApiError as err:
                st.error(str(err))
            st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- main area: chat ----------
st.title("Ask your documents")
st.caption("Answers come only from the PDFs you upload, with the page they were found on.")

if not documents:
    st.info("Upload a PDF in the sidebar to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        show_sources(message.get("sources", []))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                result = client.ask(question)
                answer, sources = result["answer"], result["sources"]
            except ApiError as err:
                answer, sources = f"Sorry, something went wrong: {err}", []
        st.write(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
