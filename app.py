"""
Lawgorithm — Intelligent Legal Document Assistant

A professional Streamlit application for AI-powered legal document analysis.
Uses LangGraph for self-correcting RAG, Groq for LLM inference, and Pinecone
for vector storage. Features real-time AI reasoning traces, risk analysis,
document intelligence, and beautiful visualization of the self-correction process.
"""

import streamlit as st
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Any

from ingestion import DocumentLoader, DocumentChunker, PineconeEmbedder
from graph import run_graph


# ============================================================
# SECTION 1: PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lawgorithm — Intelligent Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SECTION 2: CUSTOM CSS
# ============================================================

custom_css = """
<style>
    .stApp {
        background-color: #0a0a0f;
        color: #e8e8e8;
    }
    .main { background-color: #0a0a0f; }
    [data-testid="stSidebar"] {
        background-color: #0f0f15;
        border-right: 1px solid #c9a84c;
    }
    .law-header {
        background: linear-gradient(135deg, #c9a84c 0%, #d4b960 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(201, 168, 76, 0.15);
    }
    .law-header-title {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        color: #0a0a0f;
        letter-spacing: -1px;
    }
    .law-header-subtitle {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        color: rgba(10, 10, 15, 0.8);
        font-weight: 500;
    }
    .gold-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #c9a84c, transparent);
        margin: 1.5rem 0;
        border: none;
    }
    .card {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2a2a3e;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c9a84c;
        margin: 0 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #c9a84c;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #c9a84c 0%, #d4b960 100%);
        color: #0a0a0f;
        padding: 1.25rem 1.75rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        margin-left: auto;
        width: fit-content;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(201, 168, 76, 0.2);
        font-weight: 500;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .chat-bubble-assistant {
        background-color: #1a1a2e;
        color: #e8e8e8;
        padding: 1.25rem 1.75rem;
        border-radius: 16px;
        border-left: 4px solid #c9a84c;
        margin: 0.75rem 0;
        margin-right: auto;
        width: fit-content;
        max-width: 90%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .badge-high-risk {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
    .badge-medium-risk {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
    .badge-low-risk {
        background-color: #22c55e;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
    .badge-correction {
        background-color: #c9a84c;
        color: #0a0a0f;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .source-chip {
        background-color: #2a2a3e;
        color: #c9a84c;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
        border: 1px solid #c9a84c;
    }
    .trace-step {
        background-color: #1a1a2e;
        border-left: 4px solid #8a8a8a;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
    }
    .trace-step-warning {
        border-left-color: #f59e0b;
        background-color: rgba(245, 158, 11, 0.08);
    }
    .trace-step-error {
        border-left-color: #ef4444;
        background-color: rgba(239, 68, 68, 0.08);
    }
    .trace-step-success {
        border-left-color: #22c55e;
        background-color: rgba(34, 197, 94, 0.08);
    }
    .trace-step-badge {
        display: inline-block;
        background-color: #c9a84c;
        color: #0a0a0f;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .stat-card {
        background-color: #1a1a2e;
        border: 1px solid #2a2a3e;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .stat-value {
        font-size: 1rem;
        font-weight: 700;
        color: #c9a84c;
        word-break: break-word;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #8a8a8a;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .doc-card {
        background-color: #1a1a2e;
        border-left: 4px solid #c9a84c;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    .doc-card-name {
        font-weight: 600;
        color: #c9a84c;
        margin-bottom: 0.25rem;
    }
    .doc-card-info {
        color: #8a8a8a;
        font-size: 0.8rem;
        margin: 0.2rem 0;
    }
    .stButton > button {
        background-color: #1a1a2e;
        border: 2px solid #c9a84c;
        color: #c9a84c;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #c9a84c;
        color: #0a0a0f;
    }
    .welcome-card {
        background-color: #1a1a2e;
        border: 1px solid #2a2a3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .welcome-card-icon { font-size: 2.5rem; margin-bottom: 1rem; }
    .welcome-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c9a84c;
        margin-bottom: 0.5rem;
    }
    .welcome-card-desc { font-size: 0.85rem; color: #8a8a8a; }
    .footer {
        text-align: center;
        color: #c9a84c;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid #c9a84c;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# ============================================================
# SECTION 3: SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "current_namespace" not in st.session_state:
    st.session_state.current_namespace = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "correction_logs" not in st.session_state:
    st.session_state.correction_logs = []

if "last_result" not in st.session_state:
    st.session_state.last_result = {}

if "document_stats" not in st.session_state:
    st.session_state.document_stats = {}


# ============================================================
# SECTION 4: SIDEBAR
# ============================================================

with st.sidebar:

    # Sidebar header
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:2rem;'>
            <div style='font-size:3rem;'>⚖️</div>
            <div style='font-size:1.5rem; font-weight:900; color:#c9a84c;'>LAWGORITHM</div>
            <div style='font-size:0.9rem; color:#8a8a8a; margin-top:0.5rem;'>AI-Powered Legal Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    # Advanced settings
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Higher values force more rigorous evaluation and retry loops."
    )
    st.session_state.confidence_threshold = confidence_threshold

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    # Document upload section
    st.markdown("### 📂 Upload Legal Documents")

    uploaded_files = st.file_uploader(
        label="Select PDF files",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.markdown("##### Or paste legal text directly")
    pasted_text = st.text_area(
        label="Paste legal text",
        placeholder="Paste contract text, terms of service, legal agreements...",
        height=150,
        label_visibility="collapsed",
    )

    # Only show document name input if text is pasted
    doc_name = ""
    if pasted_text:
        doc_name = st.text_input(
            label="Document name",
            placeholder="e.g. Employment Contract 2024",
            label_visibility="collapsed",
        )

    # Process documents button
    if st.button("⚡ Process Documents", use_container_width=True):
        if not uploaded_files and not pasted_text:
            st.error("❌ Please upload PDF files or paste legal text")
        else:
            st.session_state.is_processing = True

            try:
                loader = DocumentLoader()
                loaded_docs = []

                my_bar = st.progress(0, text="Initializing processing...")

                # Step 1: Load PDF files using NamedTemporaryFile
                if uploaded_files:
                    my_bar.progress(10, text=f"📥 Loading {len(uploaded_files)} PDF file(s)...")
                    for uploaded_file in uploaded_files:
                        # Write to temp file then load — avoids temp dir closing too early
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name
                        try:
                            docs = loader.load_pdf(tmp_path)
                            loaded_docs.extend(docs)
                        except Exception as e:
                            st.warning(f"⚠️ Could not load {uploaded_file.name}: {str(e)}")
                        finally:
                            # Always delete temp file after loading
                            os.unlink(tmp_path)

                # Step 2: Load pasted text if provided
                if pasted_text:
                    my_bar.progress(30, text="📥 Loading pasted text...")
                    name = doc_name or "Pasted Document"
                    text_docs = loader.load_text(pasted_text, name)
                    loaded_docs.extend(text_docs)

                if not loaded_docs:
                    st.error("❌ No documents could be loaded")
                    st.session_state.is_processing = False
                    st.stop()

                # Step 3: Chunk documents
                my_bar.progress(50, text="✂️ Creating chunks...")
                chunker = DocumentChunker()
                chunks = chunker.chunk_documents(loaded_docs)

                # Step 4: Embed and store in Pinecone
                my_bar.progress(70, text="🔮 Generating embeddings and storing...")
                embedder = PineconeEmbedder()

                # Create namespace from first document name or timestamp
                if uploaded_files:
                    first_name = uploaded_files[0].name
                else:
                    first_name = doc_name or "pasted_doc"
                namespace = embedder.namespace_from_filename(first_name)

                stored_count = embedder.embed_and_store(chunks, namespace)

                # Update session state with processed document info
                st.session_state.current_namespace = namespace

                for f in (uploaded_files or []):
                    st.session_state.uploaded_documents.append({
                        "name": f.name,
                        "type": "PDF",
                        "timestamp": datetime.now().isoformat(),
                        "chunks": len(chunks),
                    })

                if pasted_text:
                    st.session_state.uploaded_documents.append({
                        "name": doc_name or "Pasted Document",
                        "type": "TEXT",
                        "timestamp": datetime.now().isoformat(),
                        "chunks": len(chunks),
                    })

                my_bar.progress(100, text=f"✨ Done! {len(loaded_docs)} pages → {len(chunks)} chunks → {stored_count} vectors")

                st.session_state.is_processing = False
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                st.session_state.is_processing = False

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    # Loaded documents list
    if st.session_state.uploaded_documents:
        st.markdown("### 📋 Loaded Documents")
        for doc in st.session_state.uploaded_documents:
            st.markdown(
                f"""
                <div class='doc-card'>
                    <div class='doc-card-name'>📄 {doc['name']}</div>
                    <div class='doc-card-info'>Type: {doc['type']}</div>
                    <div class='doc-card-info'>Chunks: {doc['chunks']}</div>
                    <div class='doc-card-info'>Added: {doc['timestamp'][:10]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    # Example questions
    st.markdown("### 💡 Example Questions")
    st.markdown(
        """
        - Summarize this contract
        - Flag all risky clauses
        - Explain the termination clause
        - What are my obligations?
        - Extract all deadlines
        - Is this contract favorable to me?
        """
    )


# ============================================================
# SECTION 5: MAIN INTERFACE
# ============================================================

# Main header
st.markdown(
    """
    <div class='law-header'>
        <div class='law-header-title'>⚖️ LAWGORITHM</div>
        <div class='law-header-subtitle'>Intelligent Legal Document Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Welcome screen if no documents loaded
if not st.session_state.current_namespace:
    st.markdown(
        """
        <div style='text-align:center; padding:3rem 1rem;'>
            <div style='font-size:4rem; margin-bottom:1rem;'>⚖️</div>
            <h1 style='color:#c9a84c; font-size:2.5rem; margin-bottom:0.5rem;'>
                Welcome to Lawgorithm
            </h1>
            <p style='color:#8a8a8a; font-size:1.1rem; margin-bottom:3rem;'>
                Upload legal documents to begin AI-powered analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🔍", "Smart Retrieval", "Pinecone vector search for accurate answers"),
        ("🛡️", "Hallucination Detection", "Every answer verified against source documents"),
        ("🔄", "Self-Correction", "Automatic refinement and validation loops"),
        ("📊", "Risk Analysis", "Flag risky clauses and unfavorable terms"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(
                f"""
                <div class='welcome-card'>
                    <div class='welcome-card-icon'>{icon}</div>
                    <div class='welcome-card-title'>{title}</div>
                    <div class='welcome-card-desc'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:
    # Three column layout
    left_col, middle_col, right_col = st.columns([1, 2.5, 1])

    # ============================================================
    # LEFT COLUMN — Document Intelligence Panel
    # ============================================================

    with left_col:
        st.markdown("### 📊 Document Analysis")

        # Document overview card
        if st.session_state.uploaded_documents:
            total_docs = len(st.session_state.uploaded_documents)
            total_chunks = sum(
                d.get("chunks", 0) for d in st.session_state.uploaded_documents
            )
            st.markdown(
                f"""
                <div class='card'>
                    <div class='card-title'>📋 Overview</div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:1rem;'>
                        <div>
                            <div style='font-size:1.8rem; font-weight:700; color:#c9a84c;'>{total_docs}</div>
                            <div style='font-size:0.75rem; color:#8a8a8a; text-transform:uppercase;'>Docs</div>
                        </div>
                        <div>
                            <div style='font-size:1.8rem; font-weight:700; color:#c9a84c;'>{total_chunks}</div>
                            <div style='font-size:0.75rem; color:#8a8a8a; text-transform:uppercase;'>Chunks</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Quick action buttons
        st.markdown("### ⚡ Quick Actions")

        if st.button("📋 Summarize Document", use_container_width=True):
            st.session_state.pending_action = "Please provide a comprehensive summary of this document."
            st.rerun()

        if st.button("🔍 Extract Deadlines", use_container_width=True):
            st.session_state.pending_action = "Extract all deadlines and important dates from this document."
            st.rerun()

        if st.button("⚖️ Analyze Favorability", use_container_width=True):
            st.session_state.pending_action = "Is this contract favorable to me? What are the main risks?"
            st.rerun()

        if st.button("🚩 Flag Risky Clauses", use_container_width=True):
            st.session_state.pending_action = "Flag all risky or unfavorable clauses in this document."
            st.rerun()

    # ============================================================
    # MIDDLE COLUMN — Chat Interface
    # ============================================================

    with middle_col:
        st.markdown("### 💬 Legal Document Conversation")

        # Scrollable chat container
        chat_container = st.container(height=500, border=True)

        with chat_container:
            for message in st.session_state.messages:
                role = message.get("role", "user")
                content = message.get("content", "")

                if role == "user":
                    # User message — gold bubble right aligned
                    st.markdown(
                        f"""
                        <div style='display:flex; justify-content:flex-end; margin-bottom:1rem;'>
                            <div class='chat-bubble-user'>{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Assistant message — dark card left aligned
                    source_files = message.get("source_files", [])
                    loop_count = message.get("loop_count", 0)
                    risk_flags = message.get("risk_flags", [])

                    score = message.get("confidence_score", None)
                    relevance = message.get("relevance_score", None)

                    # Main answer bubble
                    st.markdown(
                        f"""
                        <div style='display:flex; justify-content:flex-start; margin-bottom:0.25rem;'>
                            <div class='chat-bubble-assistant'>{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Start a div for the metadata beneath the chat bubble
                    st.markdown("<div style='margin-left: 1rem; margin-bottom: 2rem; display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;'>", unsafe_allow_html=True)

                    # Source citations and subtle score
                    src_text = ", ".join(source_files) if source_files else "None"
                    
                    score_text = ""
                    if score is not None and relevance is not None:
                        score_text = f"<span style='color:#8a8a8a; margin-left: 0.5rem;'>• Confidence: {score * 100:.0f}%</span>"

                    st.markdown(
                        f"""
                        <div style='display:inline-flex; align-items: center; font-size:0.8rem; color:#8a8a8a; 
                                    background-color:#1a1a2e; padding:0.4rem 1rem; border-radius:20px; border:1px solid #2a2a3e;'>
                            📚 Source Docs: <span style='color:#c9a84c; margin-left: 0.3rem;'>{src_text}</span>
                            {score_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Self-correction badge
                    if loop_count and loop_count > 0:
                        st.markdown(
                            f"<div class='badge-correction'>"
                            f"🔄 Self-corrected {loop_count} "
                            f"time{'s' if loop_count > 1 else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

        # Chat input at the bottom
        chat_input_val = st.chat_input(
            placeholder="Ask anything about your legal documents...",
            disabled=st.session_state.is_processing,
        )

        user_question = chat_input_val
        if st.session_state.get("pending_action"):
            user_question = st.session_state.pending_action
            st.session_state.pending_action = None

        # ============================================================
        # SECTION 6: HANDLE CHAT INPUT
        # ============================================================

        if user_question:
            # Add user message to display
            st.session_state.messages.append({
                "role": "user",
                "content": user_question,
            })

            try:
                st.session_state.is_processing = True

                with st.spinner("🤔 Analyzing your legal documents..."):
                    # Call the graph with question, namespace, and chat history
                    result = run_graph(
                        question=user_question,
                        namespace=st.session_state.current_namespace,
                        chat_history=st.session_state.chat_history,
                        confidence_threshold=st.session_state.confidence_threshold,
                    )

                # Extract all results from graph output
                generation = result.get("generation", "")
                source_files = result.get("source_files", [])
                source_pages = result.get("source_pages", [])
                loop_count = result.get("loop_count", 0)
                correction_log = result.get("correction_log", [])

                # Fallback if generation is empty
                if not generation or not generation.strip():
                    generation = (
                        "I could not find a relevant answer in the uploaded documents. "
                        "Please try rephrasing your question."
                    )

                # Store results in session state
                st.session_state.last_result = result
                st.session_state.correction_logs = correction_log

                # Update chat history for memory
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": generation,
                })

                # Add assistant message to display
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": generation,
                    "source_files": source_files,
                    "source_pages": source_pages,
                    "loop_count": loop_count,
                    "confidence_score": result.get("confidence_score", None),
                    "relevance_score": result.get("relevance_score", None),
                })

                st.session_state.is_processing = False
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.session_state.is_processing = False

    # ============================================================
    # RIGHT COLUMN — AI Reasoning Trace
    # ============================================================

    with right_col:
        st.markdown("### 📊 Backend Metrics")
        st.markdown(
            "<div style='font-size:0.8rem; color:#8a8a8a; margin-bottom:1rem;'>"
            "Confidence scores and validation metrics"
            "</div>",
            unsafe_allow_html=True,
        )

        if not st.session_state.last_result:
            # Placeholder before first query
            st.markdown(
                """
                <div style='color:#8a8a8a; font-size:0.85rem; padding:1rem;
                background-color:#1a1a2e; border-radius:8px; margin-top:1rem;'>
                    This panel displays the underlying metrics and scores for the AI's generated response. Ask a question to see the results.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.session_state.last_result:
                confidence_score = st.session_state.last_result.get("confidence_score", 0.0)
                hallucination_score = st.session_state.last_result.get("hallucination_score", 0.0)
                relevance_score = st.session_state.last_result.get("relevance_score", 0.0)
                loop_count = st.session_state.last_result.get("loop_count", 0)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<div class='stat-card'>"
                        f"<div class='stat-value'>{int(confidence_score * 100)}%</div>"
                        f"<div class='stat-label'>Confidence</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"<div class='stat-card'>"
                        f"<div class='stat-value'>{int(hallucination_score * 100)}%</div>"
                        f"<div class='stat-label'>Grounding</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(
                        f"<div class='stat-card'>"
                        f"<div class='stat-value'>{int(relevance_score * 100)}%</div>"
                        f"<div class='stat-label'>Relevance</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col4:
                    st.markdown(
                        f"<div class='stat-card'>"
                        f"<div class='stat-value'>{loop_count}</div>"
                        f"<div class='stat-label'>Loops</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Final verdict
            if st.session_state.last_result:
                lc = st.session_state.last_result.get("loop_count", 0)
                verdict = "✅ Verified" if lc == 0 else f"🔄 Corrected {lc}x"
                color = "#22c55e" if lc == 0 else "#c9a84c"
                st.markdown(
                    f"<div style='text-align:center; margin-top:1rem; "
                    f"font-size:1rem; font-weight:700; color:{color};'>"
                    f"{verdict}</div>",
                    unsafe_allow_html=True,
                )


# ============================================================
# SECTION 7: FOOTER
# ============================================================

st.markdown(
    """
    <div class='footer'>
        ⚖️ Lawgorithm v1.0<br>
        Powered by LangGraph • Gemini• Pinecone<br><br>
        <span style='font-size:0.8rem; color:#8a8a8a;'>
            For educational purposes only.
            Not a substitute for professional legal advice.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)