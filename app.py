"""
Streamlit entry point for CodeMind application.

This module sets up the Streamlit UI for CodeMind, a graph-orchestrated
agentic self-correcting RAG system for code documentation. It provides
a sidebar for repository ingestion and a chat interface for querying
the codebase.
"""

import streamlit as st  # streamlit for web UI
from typing import List, Dict, Any  # type hints

# import ingestion pipeline components
from ingestion import GitHubLoader, DocumentChunker, PineconeEmbedder

# import graph execution
from graph import run_graph


# ============================================================
# SECTION 1: PAGE CONFIGURATION
# ============================================================

# configure the page with title, icon, layout, and sidebar state
st.set_page_config(
    page_title="CodeMind — Code Documentation Assistant",  # browser tab title
    page_icon="🧠",  # emoji icon in browser tab
    layout="wide",  # wide layout for more space
    initial_sidebar_state="expanded",  # start with sidebar open
)


# ============================================================
# SECTION 2: CUSTOM CSS STYLING
# ============================================================

# apply custom CSS to make the app look professional and modern
custom_css = """
<style>
    /* main container styling */
    .main {
        padding: 0rem 0rem;
    }
    
    /* header styling with gradient background */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* chat message styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.75rem;
        display: flex;
        gap: 0.5rem;
    }
    
    /* user message: blue, right-aligned */
    .user-message {
        background-color: #3b82f6;
        color: white;
        justify-content: flex-end;
        margin-left: 20%;
    }
    
    /* assistant message: dark gray, left-aligned */
    .assistant-message {
        background-color: #1f2937;
        color: #f3f4f6;
        justify-content: flex-start;
        margin-right: 20%;
    }
    
    /* message content styling */
    .message-content {
        padding: 1rem;
        border-radius: 0.5rem;
        max-width: 100%;
    }
    
    /* sidebar styling */
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    
    /* card styling for repos */
    .repo-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* badge styling for loop count */
    .correction-badge {
        background-color: #fcd34d;
        color: #000;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
        display: inline-block;
    }
    
    /* source files styling */
    .sources-section {
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid #4b5563;
        font-size: 0.875rem;
        opacity: 0.8;
    }
</style>
"""

# inject custom CSS into the page
st.markdown(custom_css, unsafe_allow_html=True)


# ============================================================
# SECTION 3: SESSION STATE INITIALIZATION
# ============================================================

# initialize session state variables if they do not exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of chat messages (user and assistant)

if "processed_repos" not in st.session_state:
    st.session_state.processed_repos = []  # list of ingested GitHub URLs

if "current_namespace" not in st.session_state:
    st.session_state.current_namespace = ""  # current repo namespace in Pinecone

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False  # flag to prevent concurrent processing

if "current_repo_url" not in st.session_state:
    st.session_state.current_repo_url = ""  # the currently selected repo URL


# ============================================================
# SECTION 4: SIDEBAR REPOSITORY MANAGEMENT
# ============================================================

# begin the sidebar section
with st.sidebar:
    # header for the sidebar
    st.markdown("## Repository Settings")
    
    # text input for GitHub repository URL
    repo_url_input = st.text_input(
        label="GitHub Repository URL",
        placeholder="https://github.com/user/repo",
        help="Paste a GitHub repository URL to ingest its documentation",
    )
    
    # button to process the repository
    if st.button("Process Repository", use_container_width=True):
        # validate that the URL is a GitHub URL
        if not repo_url_input.startswith("https://github.com"):
            st.error("❌ URL must be a GitHub URL starting with https://github.com")
        elif repo_url_input in st.session_state.processed_repos:
            st.warning("⚠️ This repository has already been processed")
        else:
            # set processing flag to prevent concurrent operations
            st.session_state.is_processing = True
            
            try:
                # step 1: fetch files from GitHub using GitHubLoader
                with st.spinner("📥 Fetching repository files..."):
                    loader = GitHubLoader(repo_url_input)
                    files = loader.get_all_files()
                    st.success(f"✅ Fetched {len(files)} files from repository")
                
                # step 2: chunk the files using DocumentChunker
                with st.spinner(f"✂️ Chunking {len(files)} files..."):
                    chunker = DocumentChunker()
                    chunks = chunker.chunk_documents(files)
                    st.success(f"✅ Created {len(chunks)} chunks")
                
                # step 3: embed and store chunks in Pinecone using PineconeEmbedder
                with st.spinner(f"🔌 Embedding and storing {len(chunks)} chunks..."):
                    embedder = PineconeEmbedder()
                    namespace = embedder.namespace_from_url(repo_url_input)
                    stored_count = embedder.embed_and_store(chunks, namespace)
                    st.success(f"✅ Stored {stored_count} vectors in Pinecone")
                
                # add URL to processed repos list and set it as current
                st.session_state.processed_repos.append(repo_url_input)
                st.session_state.current_namespace = namespace
                st.session_state.current_repo_url = repo_url_input
                
                # success message with summary
                st.success(
                    f"✨ Repository ingested successfully!\n\n"
                    f"**Files:** {len(files)}\n"
                    f"**Chunks:** {len(chunks)}\n"
                    f"**Vectors stored:** {stored_count}"
                )
                
                # clear the input field after successful processing
                st.session_state.is_processing = False
                st.rerun()
                
            except Exception as e:
                # handle any errors gracefully with detailed message
                st.error(f"❌ Error processing repository: {str(e)}")
                st.session_state.is_processing = False
    
    # divider between sections
    st.divider()
    
    # section showing already processed repositories
    if st.session_state.processed_repos:
        st.markdown("### Processed Repositories")
        for repo in st.session_state.processed_repos:
            # show checkmark if this is the currently active repo
            is_current = repo == st.session_state.current_repo_url
            check = "✅" if is_current else "  "
            
            # clickable button to switch to this repository
            if st.button(f"{check} {repo}", use_container_width=True):
                st.session_state.current_repo_url = repo
                embedder = PineconeEmbedder()
                st.session_state.current_namespace = embedder.namespace_from_url(repo)
                st.rerun()
    
    # divider between sections
    st.divider()
    
    # section with instructions on how to use CodeMind
    st.markdown("### 📖 How to Use CodeMind")
    st.markdown(
        """
        1. **Paste Repository URL** → Copy the GitHub URL of a public repository
        2. **Process Repository** → Click the button to ingest files into the vector database
        3. **Ask Questions** → Chat naturally about the codebase
        """
    )


# ============================================================
# SECTION 5: MAIN CHAT INTERFACE
# ============================================================

# header section with gradient background and repository info
header_html = """
<div class="header-container">
    <div class="header-title">🧠 CodeMind</div>
    <div class="header-subtitle">Intelligent Code Documentation Assistant</div>
"""

# show which repository is currently loaded if one is selected
if st.session_state.current_repo_url:
    header_html += f"<div style='margin-top: 0.5rem; opacity: 0.9;'>📦 Repository: {st.session_state.current_repo_url.split('/')[-1]}</div>"
else:
    header_html += "<div style='margin-top: 0.5rem; opacity: 0.9;'>📦 No repository loaded</div>"

header_html += "</div>"
st.markdown(header_html, unsafe_allow_html=True)

# check if a repository is currently loaded
if not st.session_state.current_repo_url:
    # show welcome screen if no repository has been processed
    st.info(
        """
        👋 Welcome to CodeMind!
        
        CodeMind is an intelligent assistant that understands your codebase. 
        To get started:
        
        1. **Paste a GitHub repository URL** in the sidebar (e.g., https://github.com/tiangolo/fastapi)
        2. **Click "Process Repository"** to ingest the documentation
        3. **Ask questions** about the code — CodeMind will find answers in the codebase
        
        ### Example Questions You Could Ask:
        - "How does authentication work in this codebase?"
        - "What is the main entry point of the application?"
        - "How do I initialize this module?"
        - "What dependencies does this project have?"
        - "Explain how the API routing works"
        """
    )
else:
    # display chat history if a repository is loaded
    # create a container for scrollable chat history
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        # iterate through all messages in the chat history
        for message in st.session_state.messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                # user message: blue bubble, right-aligned
                st.markdown(
                    f"""
                    <div class="chat-message user-message">
                        <div class="message-content">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # assistant message: dark bubble, left-aligned
                source_files = message.get("source_files", [])
                loop_count = message.get("loop_count", 0)
                
                # format the assistant message content
                message_html = f"""
                <div class="chat-message assistant-message">
                    <div class="message-content">{content}
                """
                
                # add sources section if available
                if source_files:
                    message_html += f"""
                    <div class="sources-section">
                        📚 <strong>Sources:</strong> {', '.join(source_files)}
                    </div>
                    """
                
                # add self-correction badge if the answer was corrected
                if loop_count and loop_count > 0:
                    message_html += f"""
                    <div class="correction-badge">
                        🔄 Self-corrected {loop_count} time{'s' if loop_count > 1 else ''}
                    </div>
                    """
                
                message_html += """
                    </div>
                </div>
                """
                
                st.markdown(message_html, unsafe_allow_html=True)
    
    # chat input at the bottom for user questions
    user_question = st.chat_input(
        placeholder="Ask a question about the codebase...",
        disabled=st.session_state.is_processing,
    )
    
    # handle user input
    if user_question:
        # add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_question,
        })
        
        try:
            # show thinking spinner while the graph processes
            with st.spinner("🤔 Thinking..."):
                # call the graph with the question and current namespace
                result = run_graph(user_question, st.session_state.current_namespace)
            
            # extract results from the graph output
            generation = result.get("generation", "")
            source_files = result.get("source_files", [])
            loop_count = result.get("loop_count", 0)
            
            # check if generation is empty
            if not generation or generation.strip() == "":
                generation = "I could not find an answer to your question in the documentation. Please try rephrasing your question."
            
            # add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": generation,
                "source_files": source_files,
                "loop_count": loop_count,
            })
            
            # rerun to display new message
            st.rerun()
            
        except Exception as e:
            # handle any errors during graph execution
            st.error(f"❌ Error processing your question: {str(e)}")


# ============================================================
# SECTION 6: FOOTER
# ============================================================

# add a footer with helpful information
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
        CodeMind v1.0 • Built with LangGraph, Groq, and Pinecone 🚀
    </div>
    """,
    unsafe_allow_html=True,
)
