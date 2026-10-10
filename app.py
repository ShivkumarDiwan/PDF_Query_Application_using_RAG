import os
import streamlit as st
from dotenv import load_dotenv

from rag_chain import (
    extract_documents_from_pdfs,
    split_documents,
    create_vector_store,
    run_rag_pipeline,
)

# Load environment variables from .env
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Intelligence System",
    page_icon="📄",
    layout="wide",
)

# Modern Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Page Header */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .page-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 24px;
    }

    /* Card styling */
    .info-card {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if "active_page" not in st.session_state:
    st.session_state.active_page = "Upload & Analyze"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "file_names" not in st.session_state:
    st.session_state.file_names = []

# Left Sidebar Navigation
with st.sidebar:
    st.markdown("## 📑 PDF Assistant")
    st.caption("AI-Powered Document Intelligence")
    st.markdown("---")

    # Navigation menu
    menu_options = ["📁 Upload & Analyze", "💬 Document Q&A"]
    current_index = 0 if st.session_state.active_page == "Upload & Analyze" else 1

    selected_page = st.radio(
        "Navigation",
        options=menu_options,
        index=current_index,
        label_visibility="collapsed",
    )

    # Sync state if user clicked sidebar radio
    target_page = "Upload & Analyze" if "Upload" in selected_page else "Document Q&A"
    if target_page != st.session_state.active_page:
        st.session_state.active_page = target_page
        st.rerun()

    st.markdown("---")

    # Document Status & Reset Option
    if st.session_state.vector_store is not None:
        st.markdown("### 📊 Document Info")
        for fn in st.session_state.file_names:
            st.markdown(f"• **{fn}**")
        st.caption(f"{st.session_state.doc_count} pages • {st.session_state.chunk_count} text chunks")
        
        st.markdown(" ")
        if st.button("🔄 Upload New Document", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.session_state.file_names = []
            st.session_state.active_page = "Upload & Analyze"
            st.rerun()


# ==========================================
# PAGE 1: UPLOAD & ANALYZE
# ==========================================
if st.session_state.active_page == "Upload & Analyze":
    st.markdown('<div class="page-title">📄 Upload & Analyze Document</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload your PDF file to analyze its content and start asking questions.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to index.",
    )

    if uploaded_files:
        st.write(f"📁 Selected **{len(uploaded_files)} file(s)**")
        
        if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
            with st.spinner("Analyzing and indexing PDF document..."):
                try:
                    # Extract text
                    raw_docs = extract_documents_from_pdfs(uploaded_files)
                    if not raw_docs:
                        st.error("❌ No readable text found in the uploaded PDF.")
                    else:
                        # Split & Index
                        chunks = split_documents(raw_docs, chunk_size=1000, chunk_overlap=200)
                        vector_store = create_vector_store(chunks)

                        # Save to session
                        st.session_state.vector_store = vector_store
                        st.session_state.doc_count = len(raw_docs)
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.file_names = [f.name for f in uploaded_files]
                        
                        # Automatically transition to Q&A page
                        st.session_state.active_page = "Document Q&A"
                        st.rerun()

                except Exception as e:
                    err_msg = str(e)
                    if "401" in err_msg or "Incorrect API key" in err_msg:
                        st.error("❌ Invalid or missing OpenAI API Key in `.env`. Please check your `.env` file.")
                    else:
                        st.error(f"❌ Error processing document: {err_msg}")
    else:
        if st.session_state.vector_store is not None:
            st.info(f"💡 Document **{st.session_state.file_names[0]}** is already analyzed. You can switch to **Document Q&A** or upload a new file.")


# ==========================================
# PAGE 2: DOCUMENT Q&A
# ==========================================
elif st.session_state.active_page == "Document Q&A":
    st.markdown('<div class="page-title">💬 Document Q&A</div>', unsafe_allow_html=True)
    
    if st.session_state.vector_store is None:
        st.warning("⚠️ No document has been analyzed yet.")
        if st.button("👈 Go to Upload Page"):
            st.session_state.active_page = "Upload & Analyze"
            st.rerun()
    else:
        st.markdown(f'<div class="page-subtitle">Asking questions from <b>{", ".join(st.session_state.file_names)}</b> ({st.session_state.doc_count} pages)</div>', unsafe_allow_html=True)

        # Render conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📚 Sources & Citations ({len(msg['sources'])})"):
                        for idx, doc in enumerate(msg["sources"], 1):
                            src = doc.metadata.get("source", "Document")
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**Source {idx}: {src} (Page {page})**")
                            st.caption(doc.page_content)

        # Chat Input
        user_prompt = st.chat_input("Ask any question about the analyzed PDF...")

        if user_prompt:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Generate Assistant Answer
            with st.chat_message("assistant"):
                with st.spinner("Searching document & answering..."):
                    try:
                        response_text, retrieved_docs = run_rag_pipeline(
                            vector_store=st.session_state.vector_store,
                            query=user_prompt,
                            model_name="gpt-4o-mini",
                            temperature=0.2,
                            k=4,
                        )
                        st.markdown(response_text)

                        if retrieved_docs:
                            with st.expander(f"📚 Sources & Citations ({len(retrieved_docs)})"):
                                for idx, doc in enumerate(retrieved_docs, 1):
                                    src = doc.metadata.get("source", "Document")
                                    page = doc.metadata.get("page", "?")
                                    st.markdown(f"**Source {idx}: {src} (Page {page})**")
                                    st.caption(doc.page_content)

                        # Save response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": retrieved_docs,
                        })
                    except Exception as e:
                        err_msg = str(e)
                        if "401" in err_msg or "Incorrect API key" in err_msg:
                            st.error("❌ Invalid or missing OpenAI API Key in `.env`. Please check your `.env` file.")
                        else:
                            st.error(f"❌ Error generating answer: {err_msg}")
