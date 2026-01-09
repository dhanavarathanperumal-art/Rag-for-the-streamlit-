import streamlit as st
import os
import tempfile
from typing import List, Dict
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Set page configuration
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        margin-top: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #2d5d7b;
        align-self: flex-end;
        max-width: 70%;
    }
    .bot-message {
        background-color: #1e3a5f;
        align-self: flex-start;
        max-width: 70%;
    }
    .file-uploader {
        border: 2px dashed #4B8BBE;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #4B8BBE;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "embeddings_model" not in st.session_state:
        st.session_state.embeddings_model = None

init_session_state()

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Settings")
    
    model_option = st.selectbox(
        "Select Embeddings Model",
        ["OpenAI (requires API key)", "HuggingFace (free)"],
        index=1
    )
    
    if model_option == "OpenAI (requires API key)":
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            use_openai = True
        else:
            st.warning("Please enter OpenAI API key to use OpenAI embeddings")
            use_openai = False
    else:
        use_openai = False
    
    # Chunk settings
    st.subheader("Document Processing")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="Size of text chunks for processing")
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, help="Overlap between chunks")
    
    # File uploader
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or Text files for the chatbot to process"
    )
    
    # Process button
    if st.button("🚀 Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                documents = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Load document based on file type
                        if uploaded_file.name.lower().endswith('.pdf'):
                            loader = PyPDFLoader(tmp_file_path)
                            docs = loader.load()
                        elif uploaded_file.name.lower().endswith('.txt'):
                            loader = TextLoader(tmp_file_path, encoding='utf-8')
                            docs = loader.load()
                        else:
                            continue
                        
                        documents.extend(docs)
                        st.success(f"✅ Processed {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                
                if documents:
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    chunks = text_splitter.split_documents(documents)
                    
                    # Create embeddings
                    with st.spinner("Creating embeddings..."):
                        try:
                            if use_openai and api_key:
                                embeddings = OpenAIEmbeddings(
                                    openai_api_key=api_key,
                                    model="text-embedding-3-small"
                                )
                            else:
                                # Use free HuggingFace embeddings
                                embeddings = HuggingFaceEmbeddings(
                                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                                )
                            
                            # Create vector store
                            st.session_state.vector_store = FAISS.from_documents(
                                chunks,
                                embeddings
                            )
                            
                            st.session_state.documents_processed = True
                            st.session_state.processing_complete = True
                            st.success(f"✅ Processed {len(documents)} documents into {len(chunks)} chunks")
                            
                        except Exception as e:
                            st.error(f"Error creating embeddings: {str(e)}")
                else:
                    st.error("No documents could be loaded.")
        else:
            st.warning("Please upload at least one document.")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Upload PDF/TXT documents
    2. Click 'Process Documents'
    3. Start chatting!
    
    **Features:**
    - RAG-based responses
    - Document source citation
    - Conversation memory
    - Free embeddings option
    """)

# Main content area
st.markdown('<h1 class="main-header">🤖 RAG ChatBot</h1>', unsafe_allow_html=True)

# Display status
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.documents_processed:
        st.success("✅ Documents Ready")
    else:
        st.warning("📄 Upload Documents")
with col2:
    if st.session_state.vector_store:
        st.success("🔍 Vector Store Loaded")
    else:
        st.warning("🔍 Vector Store Empty")
with col3:
    st.info(f"💬 {len(st.session_state.messages)} Messages")

# Chat interface
st.markdown('<h3 class="sub-header">💬 Chat</h3>', unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>
                {message["content"]}
                <small style="opacity: 0.7; text-align: right;">{message["timestamp"]}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
                <small style="opacity: 0.7; text-align: right;">{message["timestamp"]}</small>
            </div>
            """, unsafe_allow_html=True)

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your message:",
        height=100,
        placeholder="Ask a question about your documents...",
        key="user_input"
    )
    
    col1, col2 = st.columns([6, 1])
    with col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)

# Process user input
if submit_button and user_input:
    if not st.session_state.vector_store:
        st.error("Please upload and process documents first!")
        st.stop()
    
    # Add user message to chat
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            # Initialize LLM
            if use_openai and api_key:
                llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.7,
                    streaming=False
                )
            else:
                # Use a free alternative (you can change this to other free models)
                from langchain_community.llms import HuggingFaceHub
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 0.7, "max_length": 512}
                )
            
            # Create retrieval chain
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Add previous chat history to memory
            for msg in st.session_state.messages[:-1]:  # Exclude current message
                if msg["role"] == "user":
                    memory.chat_memory.add_user_message(msg["content"])
                else:
                    memory.chat_memory.add_ai_message(msg["content"])
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
            
            # Get response
            result = qa_chain({"question": user_input})
            
            # Format response with sources
            response = result["answer"]
            
            # Add sources if available
            if "source_documents" in result:
                sources = []
                for i, doc in enumerate(result["source_documents"][:2], 1):
                    source_info = f"Source {i}: "
                    if doc.metadata.get("source"):
                        filename = doc.metadata["source"].split("/")[-1]
                        source_info += f"{filename}"
                    if doc.metadata.get("page"):
                        source_info += f" (Page {doc.metadata['page']})"
                    sources.append(source_info)
                
                if sources:
                    response += "\n\n**Sources:**\n" + "\n".join(sources)
            
            # Add bot response to chat
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp
            })
            
            # Rerun to display new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Information tabs
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 About", "🔧 How It Works", "📋 Document Info"])

with tab1:
    st.markdown("""
    ### RAG ChatBot Application
    
    **Retrieval-Augmented Generation (RAG)** combines:
    - Document retrieval from your uploaded files
    - Language model generation for contextual responses
    
    **Features:**
    - Document upload (PDF/TXT)
    - Smart text chunking
    - Vector embeddings for semantic search
    - Conversation memory
    - Source citation
    """)

with tab2:
    st.markdown("""
    ### How It Works
    
    1. **Document Processing:**
       - Upload PDF/TXT files
       - Text extraction and splitting
       - Chunking with configurable overlap
    
    2. **Vector Embeddings:**
       - Convert text to numerical vectors
       - Store in FAISS vector database
       - Enable semantic search
    
    3. **Retrieval & Generation:**
       - User query triggers similarity search
       - Retrieve relevant document chunks
       - Generate contextual response using LLM
    
    4. **Memory:**
       - Maintains conversation history
       - Provides context-aware responses
    """)

with tab3:
    if st.session_state.vector_store:
        st.success(f"Vector store contains embeddings for document chunks")
        # You could add more detailed document information here
    else:
        st.info("No documents processed yet. Upload files to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "RAG ChatBot • Built with Streamlit & LangChain • "
    "Upload documents and start chatting!"
    "</div>",
    unsafe_allow_html=True
)
