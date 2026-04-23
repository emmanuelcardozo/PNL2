# %%

import os
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_core.language_models import BaseLLM
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import tempfile
import time

# %%

import requests
import json

def validate_hf_key(hf_key):
    if not hf_key or not hf_key.startswith("hf_"):
        raise ValueError("HuggingFace Key inválida.")

# %%
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CV RAG Chatbot",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  –  editorial / dark aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
  --bg:       #0d0d0d;
  --surface:  #161616;
  --border:   #2a2a2a;
  --accent:   #e8c547;
  --accent2:  #5e9bff;
  --text:     #e8e4dc;
  --muted:    #666;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

h1, h2, h3 {
  font-family: 'Playfair Display', serif !important;
  color: var(--accent) !important;
  letter-spacing: -0.02em;
}

.stButton > button {
  background: var(--accent) !important;
  color: #000 !important;
  border: none !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 500 !important;
  border-radius: 2px !important;
  padding: 0.5rem 1.4rem !important;
  transition: opacity .2s;
}
.stButton > button:hover { opacity: .8 !important; }

[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 4px !important;
}

.stTextInput > div > input,
.stTextArea > div > textarea {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}

.chat-msg {
  padding: 0.9rem 1.2rem;
  margin-bottom: 0.6rem;
  border-radius: 4px;
  font-size: 0.93rem;
  line-height: 1.6;
}
.chat-user {
  background: #1e1e1e;
  border-left: 3px solid var(--accent);
  color: var(--text);
}
.chat-assistant {
  background: #151f2e;
  border-left: 3px solid var(--accent2);
  color: var(--text);
}
.label-user   { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--accent);  margin-bottom:.3rem; }
.label-bot    { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--accent2); margin-bottom:.3rem; }

.status-ok  { color: #4caf50; font-family:'IBM Plex Mono',monospace; font-size:.8rem; }
.status-err { color: #f44336; font-family:'IBM Plex Mono',monospace; font-size:.8rem; }
.status-inf { color: var(--accent); font-family:'IBM Plex Mono',monospace; font-size:.8rem; }

hr { border-color: var(--border) !important; }

/* hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages"         not in st.session_state: st.session_state.messages = []
if "store"            not in st.session_state: st.session_state.store = {}
if "session_id"       not in st.session_state: st.session_state.session_id = "session_001"
if "chain_ready"      not in st.session_state: st.session_state.chain_ready = False
if "conv_chain"       not in st.session_state: st.session_state.conv_chain = None
if "ingested_names"   not in st.session_state: st.session_state.ingested_names = []

# ─────────────────────────────────────────────
# SIDEBAR – credentials & upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    hf_key        = st.text_input("HuggingFace API Key", type="password", placeholder="hf_...")
    pinecone_key  = st.text_input("Pinecone API Key",    type="password", placeholder="pcsk-...")
    index_name    = st.text_input("Pinecone Index",      value="rag-cvs")
    namespace     = st.text_input("Namespace",           value="cvs")

    st.markdown("---")
    st.markdown("### 📄 Subir CVs (máx. 2)")
    uploaded_files = st.file_uploader(
        "PDF, DOCX o TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) > 2:
        st.warning("Solo se procesarán los primeros 2 archivos.")
        uploaded_files = uploaded_files[:2]

    ingest_btn = st.button("🚀 Ingestar CVs")
    if st.session_state.ingested_names:
        st.markdown("**Documentos indexados:**")
        for n in st.session_state.ingested_names:
            st.markdown(f"<span class='status-ok'>✓ {n}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<span style='font-family:IBM Plex Mono;font-size:.7rem;color:#444'>"
        "RAG · LangChain · Pinecone · HuggingFace</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_document(uploaded_file):
    """Load an uploaded file into LangChain documents."""
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    return loader.load()


def build_chain(hf_key, pinecone_key, index_name, namespace):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key.strip()

    from langchain_community.llms import HuggingFaceHub

    llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=hf_key.strip(),
    task="text2text-generation",
    model_kwargs={
        "temperature": 0.1,
        "max_length": 512
        }
    )   
    embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        pinecone_api_key=pinecone_key,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "You are an assistant specialized in analyzing candidate CVs. "
        "Use the retrieved context to answer questions about the candidates. "
        "If the information is not in the context, say so clearly. "
        "Be concise and precise. Answer in the same language as the question.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain  = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    def get_history(sid: str) -> BaseChatMessageHistory:
        if sid not in st.session_state.store:
            st.session_state.store[sid] = ChatMessageHistory()
        return st.session_state.store[sid]

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def ingest_documents(files, hf_key, pinecone_key, index_name, namespace):
    all_docs = []
    names = []
    for f in files:
        docs = load_document(f)
        all_docs.extend(docs)
        names.append(f.name)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # USAMOS ESTE MODELO QUE ES ESTABLE EN LA API
    embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pc = Pinecone(api_key=pinecone_key)
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(index_name, dimension=384, metric="cosine", spec=spec)
    time.sleep(5)

    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
        pinecone_api_key=pinecone_key,
    )
    return names


# ─────────────────────────────────────────────
# INGEST TRIGGER
# ─────────────────────────────────────────────
if ingest_btn:
    if not hf_key or not pinecone_key:
        st.error("Completá las API Keys en la barra lateral.")
    elif not uploaded_files:
        st.warning("Subí al menos un CV.")
    else:
        with st.spinner("Procesando documentos e indexando en Pinecone…"):
            try:
#                results = validate_hf_key(hf_key)
#                st.write("🔍 Diagnóstico HF:")
#                st.json(results)   
                names = ingest_documents(
                    uploaded_files, hf_key, pinecone_key, index_name, namespace
                )
                st.session_state.ingested_names = names
                st.session_state.conv_chain     = build_chain(
                    hf_key, pinecone_key, index_name, namespace
                )
                st.session_state.chain_ready    = True
                st.session_state.messages       = []
                st.session_state.store          = {}
                st.success(f"✅ {len(names)} CV(s) indexados correctamente.")
            except Exception as e:
                st.error(f"Error durante la ingestión: {e}")

# ─────────────────────────────────────────────
# MAIN – header
# ─────────────────────────────────────────────
st.markdown("# CV RAG Chatbot")
st.markdown(
    "<p style='font-family:IBM Plex Mono;font-size:.85rem;color:#666;margin-top:-.5rem'>"
    "Retrieval-Augmented Generation · Consultá los CVs en lenguaje natural</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# ARCHITECTURE DIAGRAM (collapsible)
# ─────────────────────────────────────────────
with st.expander("📐 Ver arquitectura del sistema"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pipeline de Ingestión**")
        st.markdown("""
```
CV (PDF/DOCX/TXT)
      ↓
Document Loader
      ↓
Text Splitter / Chunker
      ↓
Embedding Model (text-embedding-3-large)
      ↓
Vector Database (Pinecone)
```
""")
    with col2:
        st.markdown("**Runtime / Question Answering**")
        st.markdown("""
```
User Question
      ↓
Embed Question
      ↓
Similarity Search in Vector DB
      ↓
Top-K Relevant Chunks
      ↓
Prompt Builder (Question + Context + History)
      ↓
LLM (gpt-3.5-turbo)
      ↓
Final Answer → Chat UI
```
""")

# ─────────────────────────────────────────────
# CHAT AREA
# ─────────────────────────────────────────────
if not st.session_state.chain_ready:
    if hf_key and pinecone_key:
        try:
            st.session_state.conv_chain  = build_chain(hf_key, pinecone_key, index_name, namespace)
            st.session_state.chain_ready = True
        except Exception:
            pass

if st.session_state.chain_ready:
    # Render existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-msg chat-user'>"
                f"<div class='label-user'>YOU</div>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-msg chat-assistant'>"
                f"<div class='label-bot'>ASSISTANT</div>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    # Input
    user_input = st.chat_input("Hacé una pregunta sobre los CVs…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(
            f"<div class='chat-msg chat-user'>"
            f"<div class='label-user'>YOU</div>{user_input}</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Buscando en los CVs…"):
            try:
                result = st.session_state.conv_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
                answer = result["answer"]
            except Exception as e:
                answer = f"Error al generar respuesta: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(
            f"<div class='chat-msg chat-assistant'>"
            f"<div class='label-bot'>ASSISTANT</div>{answer}</div>",
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.store    = {}
        st.rerun()

else:
    st.markdown(
        "<div style='text-align:center;padding:3rem;color:#444;font-family:IBM Plex Mono;'>"
        "⬅ Completá las API Keys y subí los CVs para comenzar."
        "</div>",
        unsafe_allow_html=True,
    )