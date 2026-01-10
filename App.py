import streamlit as st
import PyPDF2
import io
import os
import time
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Offline RAG ChatBot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .main-title {
        text-align: center;
        color: #60a5fa;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .offline-badge {
        display: inline-block;
        background: #10b981;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-left: 1rem;
    }
    .file-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .chat-user {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .chat-bot {
        background: linear-gradient(90deg, #0f172a, #1e293b);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<h1 class="main-title">
    🤖 Offline RAG ChatBot
    <span class="offline-badge">100% LOCAL</span>
</h1>
<p style="text-align: center; color: #94a3b8; margin-bottom: 2rem;">
    No API keys needed • Works completely offline • Private & Secure
</p>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection
    st.markdown("#### 🧠 Model Settings")
    model_choice = st.selectbox(
        "Choose Model Size",
        ["Tiny (Fast, less accurate)", "Small (Good balance)", "Medium (Better accuracy)"],
        index=1,
        help="Smaller models are faster but less accurate"
    )
    
    # Document processing
    st.markdown("#### 📄 Document Processing")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 
                          help="Size of text chunks for processing")
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200,
                             help="Overlap between chunks for context")
    
    # File upload
    st.markdown("#### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF files to chat with"
    )
    
    # Load models button
    if not st.session_state.models_loaded:
        if st.button("🔧 Load AI Models", use_container_width=True):
            with st.spinner("Downloading models (first time may take 2-3 minutes)..."):
                try:
                    # This will download the models
                    from sentence_transformers import SentenceTransformer
                    from transformers import pipeline
                    
                    # Download embedding model
                    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    st.session_state.embedding_model = embedding_model
                    
                    # Download small LLM
                    st.info("Downloading language model... This may take a moment.")
                    llm = pipeline(
                        'text-generation',
                        model='distilgpt2',  # Small model that works offline
                        max_length=200,
                        temperature=0.7
                    )
                    st.session_state.llm = llm
                    
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                    
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
                    st.info("Make sure all dependencies are installed in requirements.txt")
    else:
        st.success("✅ Models are loaded and ready!")
    
    # Process documents button
    if uploaded_files and st.session_state.models_loaded:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    documents = []
                    
                    for uploaded_file in uploaded_files:
                        # Read PDF
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                        text = ""
                        for page_num, page in enumerate(pdf_reader.pages):
                            text += f"\n--- Page {page_num+1} ---\n"
                            text += page.extract_text()
                        
                        documents.append({
                            'name': uploaded_file.name,
                            'text': text,
                            'pages': len(pdf_reader.pages)
                        })
                    
                    st.session_state.documents = documents
                    
                    # Create chunks
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    
                    all_chunks = []
                    for doc in documents:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        
                        chunks = text_splitter.split_text(doc['text'])
                        for chunk in chunks:
                            all_chunks.append({
                                'text': chunk,
                                'source': doc['name']
                            })
                    
                    # Create embeddings and vector store
                    import numpy as np
                    from sentence_transformers import SentenceTransformer
                    
                    # Get embeddings
                    model = st.session_state.embedding_model
                    chunk_texts = [chunk['text'] for chunk in all_chunks]
                    embeddings = model.encode(chunk_texts)
                    
                    # Simple vector store using FAISS
                    try:
                        import faiss
                        dimension = embeddings.shape[1]
                        index = faiss.IndexFlatL2(dimension)
                        index.add(embeddings)
                        
                        st.session_state.vector_store = {
                            'index': index,
                            'chunks': all_chunks,
                            'embeddings': embeddings
                        }
                        
                        st.success(f"✅ Processed {len(documents)} documents into {len(all_chunks)} chunks")
                        
                    except Exception as e:
                        # Fallback to simple similarity search
                        st.warning("Using simple similarity search (FAISS not available)")
                        st.session_state.vector_store = {
                            'chunks': all_chunks,
                            'embeddings': embeddings
                        }
                        st.success(f"✅ Processed {len(documents)} documents")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content area
# Display documents if processed
if st.session_state.documents:
    st.markdown("### 📚 Processed Documents")
    for doc in st.session_state.documents:
        with st.expander(f"📄 {doc['name']} ({doc['pages']} pages)"):
            st.text(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])

# Chat interface
st.markdown("### 💬 Chat with Your Documents")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-user">
            <strong>👤 You:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-bot">
            <strong>🤖 Assistant:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    if not st.session_state.models_loaded:
        st.error("Please load the AI models from the sidebar first!")
        st.stop()
    
    if not st.session_state.vector_store:
        st.error("Please process some documents first!")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            # Search for relevant chunks
            model = st.session_state.embedding_model
            query_embedding = model.encode([prompt])
            
            if 'index' in st.session_state.vector_store:
                # Use FAISS for search
                import faiss
                index = st.session_state.vector_store['index']
                D, I = index.search(query_embedding, 3)  # Get top 3 results
                
                relevant_chunks = []
                for idx in I[0]:
                    if idx < len(st.session_state.vector_store['chunks']):
                        chunk = st.session_state.vector_store['chunks'][idx]
                        relevant_chunks.append(chunk['text'])
            else:
                # Simple similarity search
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                
                embeddings = st.session_state.vector_store['embeddings']
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                relevant_chunks = []
                for idx in top_indices:
                    chunk = st.session_state.vector_store['chunks'][idx]
                    relevant_chunks.append(chunk['text'])
            
            # Prepare context
            context = "\n\n".join(relevant_chunks[:3])
            
            # Generate answer using local LLM
            llm = st.session_state.llm
            
            prompt_template = f"""Based on the following context, answer the question.
            
            Context:
            {context}
            
            Question: {prompt}
            
            Answer:"""
            
            response = llm(
                prompt_template,
                max_length=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )[0]['generated_text']
            
            # Extract just the answer part
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.strip()
            
            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Rerun to show new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            # Fallback to simple keyword-based response
            all_text = ""
            for doc in st.session_state.documents:
                all_text += doc['text'] + "\n\n"
            
            # Simple keyword matching
            keywords = prompt.lower().split()
            sentences = all_text.split('.')
            relevant = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    relevant.append(sentence.strip())
            
            if relevant:
                answer = f"I found this information: {' '.join(relevant[:3])}"
            else:
                answer = "I couldn't find specific information about that in your documents."
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

# Information section
st.markdown("---")
st.markdown("### ℹ️ How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📥 **Offline Models**
    - Uses free, downloadable models
    - No internet required after setup
    - Your data stays private
    """)

with col2:
    st.markdown("""
    #### 🔍 **Document Processing**
    - Extracts text from PDFs
    - Creates searchable chunks
    - Uses local embeddings
    """)

with col3:
    st.markdown("""
    #### 🤖 **Local AI**
    - Runs entirely on your machine
    - No API costs
    - 100% private conversations
    """)

# Quick setup instructions
with st.expander("⚡ Quick Setup Guide"):
    st.markdown("""
    1. **Click 'Load AI Models'** in the sidebar (first time takes 2-3 minutes)
    2. **Upload PDF files** you want to chat with
    3. **Click 'Process Documents'** to make them searchable
    4. **Start asking questions!**
    
    **Note:** The first time you load models, it will download them (about 300MB total).
    Subsequent runs will be much faster as models are cached.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b;'>"
    "🤖 Offline RAG ChatBot • No API Keys • 100% Local • Private & Secure"
    "</div>",
    unsafe_allow_html=True
)
