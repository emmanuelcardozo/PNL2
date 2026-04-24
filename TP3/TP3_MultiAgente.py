
# %%

import os
import re
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import tempfile
import time

# %%

import requests

def validate_openai_key(openai_key):
    r = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {openai_key}"}
    )
    if r.status_code != 200:
        raise ValueError(f"OpenAI Key inválida o sin permisos: {r.text}")

# %%
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CV RAG Multi-Agent Chatbot",
    page_icon="🤖",
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
  --accent3:  #ff6b6b;
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
.chat-system {
  background: #1a1a1a;
  border-left: 3px solid var(--accent3);
  color: var(--muted);
  font-size: 0.8rem;
  font-family: 'IBM Plex Mono', monospace;
  padding: 0.4rem 0.8rem;
}
.label-user    { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--accent);  margin-bottom:.3rem; }
.label-bot     { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--accent2); margin-bottom:.3rem; }
.label-routing { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--accent3); margin-bottom:.3rem; }

.agent-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 2px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  font-weight: 500;
  margin-right: 4px;
}
.badge-default  { background: #2a2a1a; color: var(--accent); border: 1px solid var(--accent); }
.badge-named    { background: #1a2030; color: var(--accent2); border: 1px solid var(--accent2); }
.badge-multi    { background: #2a1a1a; color: var(--accent3); border: 1px solid var(--accent3); }

.status-ok  { color: #4caf50; font-family:'IBM Plex Mono',monospace; font-size:.8rem; }
.status-err { color: #f44336; font-family:'IBM Plex Mono',monospace; font-size:.8rem; }
.status-inf { color: var(--accent); font-family:'IBM Plex Mono',monospace; font-size:.8rem; }

hr { border-color: var(--border) !important; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "messages":         [],
    "store":            {},
    "session_id":       "session_001",
    "chain_ready":      False,
    # Dict: nombre_normalizado -> RunnableWithMessageHistory
    "agents":           {},
    # Lista ordenada de nombres tal como se cargaron (el primero = agente por defecto)
    "agent_order":      [],
    # Dict: nombre_normalizado -> namespace en Pinecone
    "agent_namespaces": {},
    "ingested_names":   [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SIDEBAR – credentials & upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    openai_key   = st.text_input("OpenAI API Key",   type="password", placeholder="sk-...")
    pinecone_key = st.text_input("Pinecone API Key", type="password", placeholder="pcsk-...")
    index_name   = st.text_input("Pinecone Index",   value="rag-cvs")

    st.markdown("---")
    st.markdown("### 📄 Subir CVs (1 por persona)")
    st.markdown(
        "<span class='status-inf'>Nombrar archivos con el nombre de la persona.<br>"
        "Ej: cardozo.pdf, didone.docx</span>",
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "PDF, DOCX o TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    ingest_btn = st.button("🚀 Ingestar CVs")

    if st.session_state.agent_order:
        st.markdown("**Agentes activos:**")
        for i, name in enumerate(st.session_state.agent_order):
            badge = "🟡 (default)" if i == 0 else "🔵"
            st.markdown(
                f"<span class='status-ok'>{badge} {name.capitalize()}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        "<span style='font-family:IBM Plex Mono;font-size:.7rem;color:#444'>"
        "Multi-Agent RAG · LangChain · Pinecone · OpenAI</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize_name(filename: str) -> str:
    """Extract a clean ASCII lowercase name from a filename (safe for Pinecone namespaces).
    Strips accents: é→e, ñ→n, ü→u, etc., then keeps only alphanumerics.
    """
    import unicodedata
    name = filename.rsplit(".", 1)[0]          # remove extension
    # Decompose unicode and drop combining characters (é→e, ñ→n, ü→u …)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Keep only ASCII letters/digits/spaces
    name = re.sub(r"[^a-zA-Z0-9\s]", " ", name)
    name = name.strip().lower()
    # Use the first word as the person key ("jose_cv_2024" → "jose")
    first_word = name.split()[0] if name.split() else name
    return first_word


def load_document(uploaded_file):
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


def build_agent(openai_key, pinecone_key, index_name, namespace, person_name):
    """Build a RAG chain scoped to a single person's namespace."""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_key.strip(),
        temperature=0.1,
        max_tokens=512,
    )
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_key.strip(),
    )
    vectorstore = PineconeVectorStore(
        pinecone_api_key=pinecone_key,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        f"Eres un experto analizando el perfil de {person_name.capitalize()}. "
        "Responde de forma profesional y directa sobre los datos presentes en su CV. "
        "Si la pregunta te pide comparar o menciona datos que no están en su documento, "
        "limítate a resumir lo que SÍ aparece en su historial profesional sobre el tema solicitado. "
        "No intentes buscar conexiones con otras personas.\n\n"
        "Contexto: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain  = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    def get_history(sid: str) -> BaseChatMessageHistory:
        key = f"{person_name}_{sid}"
        if key not in st.session_state.store:
            st.session_state.store[key] = ChatMessageHistory()
        return st.session_state.store[key]

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def ingest_single_cv(file, openai_key, pinecone_key, index_name, namespace):
    """Ingest one CV into a specific namespace."""
    docs = load_document(file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_key.strip(),
    )
    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
        pinecone_api_key=pinecone_key,
    )


def detect_mentioned_persons(query: str, known_names: list[str]) -> list[str]:
    """
    Return the list of known person names mentioned in the query (case-insensitive).
    Empty list = nobody mentioned → use default agent.
    """
    query_lower = query.lower()
    return [name for name in known_names if name in query_lower]


# ─────────────────────────────────────────────
# ORCHESTRATOR  –  route query to agent(s)
# ─────────────────────────────────────────────

def orchestrate(query: str, session_id: str) -> tuple[str, str, str]:
    """
    Route the query and invoke the appropriate agent(s).

    Returns:
        answer  (str)  – final response text
        routing (str)  – human-readable routing decision
        mode    (str)  – 'default' | 'named' | 'multi'
    """
    agents      = st.session_state.agents
    order       = st.session_state.agent_order
    known_names = list(agents.keys())

    mentioned = detect_mentioned_persons(query, known_names)

    # ── Case 1: nobody mentioned → default agent (first loaded) ──────────────
    if not mentioned:
        default_name  = order[0]
        routing_label = f"No se mencionó ninguna persona → agente por defecto: **{default_name.capitalize()}**"
        mode          = "default"
        result = agents[default_name].invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        return result["answer"], routing_label, mode

    # ── Case 2: exactly one person mentioned ─────────────────────────────────
    if len(mentioned) == 1:
        name          = mentioned[0]
        routing_label = f"Persona detectada → agente: **{name.capitalize()}**"
        mode          = "named"
        result = agents[name].invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        return result["answer"], routing_label, mode

# ── Case 3: multiple persons mentioned ───────────────────────────────────

    mode = "multi"
    routing_label = f"Múltiples personas detectadas → agentes: **{', '.join(n.capitalize() for n in mentioned)}**"
    combined_parts = []
    
    for name in mentioned:
        # 1. Limpiamos la query: eliminamos menciones a otros candidatos 
        # para que el agente no intente buscar "junto a..." o "comparado con..."
        others = [n for n in mentioned if n != name]
        clean_query = query
        for other in others:
            # Reemplaza el nombre del otro por un espacio para evitar confusión
            clean_query = re.sub(rf"\b{other}\b", "", clean_query, flags=re.IGNORECASE)
        
        # 2. Instrucción ultra-específica y aislada
        prompt_individual = f"Analiza únicamente la información de {name.capitalize()} respecto a: {clean_query}"
        
        result = agents[name].invoke(
            {"input": prompt_individual},
            config={"configurable": {"session_id": session_id}},
        )
        
        # 3. Formateo de salida independiente
        combined_parts.append(f"### 📄 Perfil: {name.capitalize()}\n{result['answer']}")
    
    combined_answer = "\n\n---\n\n".join(combined_parts)
    return combined_answer, routing_label, mode


# ─────────────────────────────────────────────
# INGEST TRIGGER
# ─────────────────────────────────────────────
if ingest_btn:
    if not openai_key or not pinecone_key:
        st.error("Completá las API Keys en la barra lateral.")
    elif not uploaded_files:
        st.warning("Subí al menos un CV.")
    else:
        with st.spinner("Procesando documentos e indexando en Pinecone…"):
            try:
                validate_openai_key(openai_key)

                # Ensure Pinecone index exists (dimension 1536 for text-embedding-3-small)
                pc   = Pinecone(api_key=pinecone_key)
                spec = ServerlessSpec(cloud="aws", region="us-east-1")
                if index_name not in pc.list_indexes().names():
                    pc.create_index(index_name, dimension=1536, metric="cosine", spec=spec)
                    time.sleep(5)

                new_agents     = {}
                new_order      = []
                new_namespaces = {}
                ingested_names = []

                for f in uploaded_files:
                    person_key = normalize_name(f.name)
                    namespace  = f"cv_{person_key}"

                    ingest_single_cv(f, openai_key, pinecone_key, index_name, namespace)

                    agent = build_agent(openai_key, pinecone_key, index_name, namespace, person_key)

                    new_agents[person_key]     = agent
                    new_order.append(person_key)
                    new_namespaces[person_key] = namespace
                    ingested_names.append(f.name)

                st.session_state.agents           = new_agents
                st.session_state.agent_order      = new_order
                st.session_state.agent_namespaces = new_namespaces
                st.session_state.ingested_names   = ingested_names
                st.session_state.chain_ready      = True
                st.session_state.messages         = []
                st.session_state.store            = {}

                st.success(
                    f"✅ {len(ingested_names)} CV(s) indexados. "
                    f"Agente por defecto: **{new_order[0].capitalize()}**"
                )

            except Exception as e:
                st.error(f"Error durante la ingestión: {e}")

# ─────────────────────────────────────────────
# MAIN – header
# ─────────────────────────────────────────────
st.markdown("# CV RAG · Multi-Agent Chatbot")
st.markdown(
    "<p style='font-family:IBM Plex Mono;font-size:.85rem;color:#666;margin-top:-.5rem'>"
    "1 Agente por persona · Routing automático · Retrieval-Augmented Generation</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# ARCHITECTURE DIAGRAM (collapsible)
# ─────────────────────────────────────────────
with st.expander("📐 Ver arquitectura del sistema"):
    st.markdown("""
```
User Query
    ↓
Agent Orchestrator  ─────────────────────────────────────────────────────┐
    ↓                                                                     │
Routing / Logic Node                                              Conversation Memory
  ┌─────┬──────┬──────┬────────────────────┐                            │
  ▼     ▼      ▼      ▼                    ▼                             │
Alice  Bob  Carol  (otros)     Ambiguous / Multiple → combinar respuestas│
  ↓     ↓      ↓                                                         │
Retriever-Alice  Retriever-Bob  Retriever-Carol                          │
  ↓     ↓      ↓                                                         │
Alice Index  Bob Index  Carol Index  (namespaces separados)              │
  └─────┴──────┴──────┘                                                  │
         ↓                                                                │
   Single Vector DB (Pinecone)                                           │
         ↓                                                                │
   Relevant Chunks  ──────────────────→  Prompt Builder ←────────────────┘
                                               ↓
                                         LLM (gpt-3.5-turbo)
                                               ↓
                                         Final Answer → Chat UI
```
**Regla de routing:**
- Sin nombre en la query → Agente del **primer alumno** cargado (por defecto)
- Nombre detectado → Agente específico de esa persona
- Múltiples nombres → Se consultan todos los agentes y se combinan las respuestas
""")

# ─────────────────────────────────────────────
# CHAT AREA
# ─────────────────────────────────────────────

# Allow loading agents from existing index if keys present and no ingest done yet
if not st.session_state.chain_ready and openai_key and pinecone_key and st.session_state.agent_order:
    try:
        rebuilt = {}
        for name in st.session_state.agent_order:
            ns = st.session_state.agent_namespaces[name]
            rebuilt[name] = build_agent(openai_key, pinecone_key, index_name, ns, name)
        st.session_state.agents      = rebuilt
        st.session_state.chain_ready = True
    except Exception:
        pass

if st.session_state.chain_ready and st.session_state.agents:

    # ── Render existing messages ──────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-msg chat-user'>"
                f"<div class='label-user'>YOU</div>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        elif msg["role"] == "routing":
            st.markdown(
                f"<div class='chat-system'>"
                f"<div class='label-routing'>⚙ ROUTING</div>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-msg chat-assistant'>"
                f"<div class='label-bot'>ASSISTANT</div>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = st.chat_input("Hacé una pregunta sobre los CVs…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(
            f"<div class='chat-msg chat-user'>"
            f"<div class='label-user'>YOU</div>{user_input}</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Consultando agentes…"):
            try:
                answer, routing_label, mode = orchestrate(
                    user_input, st.session_state.session_id
                )
            except Exception as e:
                answer        = f"Error al generar respuesta: {e}"
                routing_label = "Error en el routing"
                mode          = "default"

        # Show routing decision
        badge_class = {"default": "badge-default", "named": "badge-named", "multi": "badge-multi"}[mode]
        routing_html = f"<span class='agent-badge {badge_class}'>{mode.upper()}</span> {routing_label}"
        st.session_state.messages.append({"role": "routing", "content": routing_html})
        st.markdown(
            f"<div class='chat-system'><div class='label-routing'>⚙ ROUTING</div>{routing_html}</div>",
            unsafe_allow_html=True,
        )

        # Show answer
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
