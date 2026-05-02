# app.py

import streamlit as st

from document_store import load_documents
from online_doc_ai import (
    has_api_key,
    get_file_signature,
    upload_files_to_gemini,
    answer_from_documents,
    summarize_documents,
    delete_gemini_files,
    DEFAULT_MODEL,
)


DEV_MODE = False
GEMINI_MODEL_NAME = DEFAULT_MODEL


st.set_page_config(
    page_title="AI Doc Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css():
    try:
        with open("style.css", "r", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def initialize_state():
    defaults = {
        "chat_history": [],
        "summary_cache": {},
        "uploaded_signature": None,
        "gemini_files": [],
        "doc_data": None,
        "show_sources": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_document_state(delete_remote_files: bool = False):
    if delete_remote_files:
        delete_gemini_files(st.session_state.get("gemini_files", []))

    for key in [
        "chat_history",
        "summary_cache",
        "uploaded_signature",
        "gemini_files",
        "doc_data",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    initialize_state()


def render_sources(sources):
    if not sources or not st.session_state.show_sources:
        return

    with st.expander("Sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            source_name = source.get("source", "Uploaded document")
            page = source.get("page", "Unknown")
            source_type = source.get("type", "Evidence")
            section = source.get("section", "Unknown")
            text = source.get("text", "")

            st.markdown(
                f"**{index}. {source_name} — Page {page} — {source_type}**"
            )

            if section and section != "Unknown":
                st.caption(f"Section: {section}")

            if text:
                st.write(text[:1500])


def render_uploaded_files(file_records):
    st.markdown("**Uploaded Files**")

    for record in file_records:
        name = record.get("name", "Uploaded file")
        size = record.get("size", 0)

        icon = "📄" if name.lower().endswith(".pdf") else "🖼️"

        st.markdown(
            f"{icon} **{name}** · {size // 1024} KB"
        )


def render_document_info(file_records, pages, chunks):
    st.markdown("**Document Info**")

    pages_count = len(pages) if pages else "Gemini-read"
    chunks_count = len(chunks) if chunks else "Gemini document mode"

    st.markdown(
        f"• **Files:** {len(file_records)} &nbsp;&nbsp; • **Pages / Images:** {pages_count}",
        unsafe_allow_html=True,
    )
    st.markdown(f"• **Chunks:** {chunks_count}")
    st.markdown("• **Status:** Ready")


load_css()
initialize_state()


st.markdown("## AI DOC ASSISTANT")


top_left, top_right = st.columns([1.45, 0.55], gap="large")


with top_left:
    st.markdown("**Upload Documents**")

    uploaded_files = st.file_uploader(
        "Upload PDF or images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.caption("Supported: PDF, PNG, JPG, JPEG")


with top_right:
    st.markdown("**Controls**")

    st.session_state.show_sources = st.checkbox(
        "Show Sources",
        value=st.session_state.show_sources,
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Reset Uploaded Files", use_container_width=True):
        reset_document_state(delete_remote_files=True)
        st.rerun()


if not has_api_key():
    st.error("Gemini API key is missing.")

    st.code(
        'Create .streamlit/secrets.toml and add:\n\nGEMINI_API_KEY = "YOUR_API_KEY"',
        language="toml",
    )

    st.stop()


if not uploaded_files:
    st.info("Upload one or more files to start chatting with your document.")
    st.stop()


current_signature = get_file_signature(uploaded_files)


if st.session_state.uploaded_signature != current_signature:
    reset_document_state(delete_remote_files=True)
    st.session_state.uploaded_signature = current_signature

    with st.spinner("Preparing document..."):
        try:
            st.session_state.doc_data = load_documents(uploaded_files)
        except Exception as error:
            st.warning(f"Local document preview failed: {error}")
            st.session_state.doc_data = None

        try:
            st.session_state.gemini_files = upload_files_to_gemini(uploaded_files)
        except Exception as error:
            st.error(f"Document upload failed: {error}")
            st.stop()

    st.session_state.chat_history = []
    st.session_state.summary_cache = {}
    st.rerun()


doc_data = st.session_state.doc_data
gemini_files = st.session_state.gemini_files


if not gemini_files:
    st.error("Document is not ready.")
    st.stop()


if doc_data:
    file_records = doc_data.get("file_records", [])
    pages = doc_data.get("pages", [])
    chunks = doc_data.get("chunks", [])
else:
    file_records = [
        {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
        }
        for uploaded_file in uploaded_files
    ]
    pages = []
    chunks = []


info_left, info_right = st.columns([1.45, 0.55], gap="large")


with info_left:
    render_uploaded_files(file_records)


with info_right:
    render_document_info(file_records, pages, chunks)


tab_chat, tab_summary = st.tabs(["💬 Chat", "📌 Summary"])


with tab_chat:
    st.markdown("### Ask your document")

    if not st.session_state.chat_history:
        st.info("Ready. Ask a question about the uploaded document.")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])

        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            render_sources(chat.get("sources", []))

    question = st.chat_input("Ask something about your document...")

    if question:
        history_for_model = [
            item for item in st.session_state.chat_history
            if item.get("answer")
        ]

        with st.spinner("Thinking from the document..."):
            result = answer_from_documents(
                question=question,
                gemini_files=gemini_files,
                chat_history=history_for_model,
                model_name=GEMINI_MODEL_NAME,
            )

        st.session_state.chat_history.append({
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        })

        st.rerun()

    if DEV_MODE and doc_data:
        debug_left, debug_right = st.columns(2)

        with debug_left:
            with st.expander("Debug: First extracted pages", expanded=False):
                for page in pages[:3]:
                    st.markdown(
                        f"**{page['source']} — Page {page['page']} — {page.get('type', 'text')}**"
                    )
                    st.write(page["text"][:2000])

        with debug_right:
            with st.expander("Debug: First chunks", expanded=False):
                for chunk in chunks[:5]:
                    st.markdown(
                        f"**Chunk ID {chunk.get('id', '-') } — {chunk['source']} — Page {chunk['page']} — {chunk.get('type', 'text')}**"
                    )
                    st.write(chunk["text"])


with tab_summary:
    st.markdown("### Summarize document")

    mode = st.selectbox(
        "Summary Type",
        [
            "Short Summary",
            "Detailed Summary",
            "Bullet Summary",
        ],
    )

    summary_cache_key = (
        str(st.session_state.uploaded_signature),
        mode,
        GEMINI_MODEL_NAME,
    )

    if st.button("Generate Summary", use_container_width=True):
        if summary_cache_key in st.session_state.summary_cache:
            summary = st.session_state.summary_cache[summary_cache_key]
        else:
            with st.spinner("Generating summary..."):
                summary = summarize_documents(
                    gemini_files=gemini_files,
                    summary_mode=mode,
                    model_name=GEMINI_MODEL_NAME,
                )

            st.session_state.summary_cache[summary_cache_key] = summary

        st.markdown(
            f'<div class="summary-box">{summary}</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download Summary",
            summary,
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True,
        )