# CV RAG · Multi-Agent Chatbot

Sistema de consulta de CVs basado en Retrieval-Augmented Generation (RAG) con arquitectura multi-agente. Cada integrante del equipo tiene su propio agente especializado; un orquestador central enruta cada consulta al agente o agentes correspondientes según los nombres mencionados en la pregunta.

**Trabajo Práctico 6 — LLMs · Cardozo & Didoné**

---

## Arquitectura

```
User Query
    ↓
Agent Orchestrator
    ↓
Routing / Logic Node  ──────────────────────────────────────────┐
  ┌──────────┬──────────┬──────────┐                            │
  ▼          ▼          ▼          ▼                    Conversation Memory
Agente-1  Agente-2  Agente-3  (múltiples)                       │
  ↓          ↓          ↓                                        │
Retriever  Retriever  Retriever                                  │
  ↓          ↓          ↓                                        │
Namespace  Namespace  Namespace  ← Pinecone (separados por CV)  │
  └──────────┴──────────┴──────────┘                            │
                  ↓                                              │
           Relevant Chunks ──────→ Prompt Builder ←─────────────┘
                                         ↓
                                   LLM (gpt-3.5-turbo)
                                         ↓
                                   Final Answer → Chat UI
```

### Reglas de routing

| Situación | Comportamiento |
|---|---|
| No se menciona ningún nombre | Se usa el **agente por defecto** (primer CV cargado) |
| Se menciona un nombre | Se invoca el agente de esa persona |
| Se mencionan varios nombres | Se invocan todos los agentes relevantes y se combinan las respuestas |

---

## Tecnologías

- **[Streamlit](https://streamlit.io/)** — interfaz de chat
- **[LangChain](https://www.langchain.com/)** — orquestación de cadenas RAG y memoria conversacional
- **[OpenAI](https://platform.openai.com/)** — embeddings (`text-embedding-3-small`) y LLM (`gpt-3.5-turbo`)
- **[Pinecone](https://www.pinecone.io/)** — base de datos vectorial con namespaces separados por persona

---

## Requisitos

- Python 3.10+
- Cuenta en [OpenAI Platform](https://platform.openai.com/) con API key
- Cuenta en [Pinecone](https://www.pinecone.io/) con API key e índice creado (dimensión `1536`, métrica `cosine`, cloud `aws`, región `us-east-1`)

### Instalación de dependencias

```bash
pip install streamlit langchain langchain-community langchain-openai langchain-pinecone pinecone-client python-docx docx2txt pypdf requests
```

---

## Cómo correr la aplicación

```bash
streamlit run TP6_Rag_MultiAgent_Cardozo-Didone.py
```

---

## Uso paso a paso

### 1. Configurar credenciales
En la barra lateral completar:
- **OpenAI API Key** — comienza con `sk-...`
- **Pinecone API Key** — comienza con `pcsk-...`
- **Pinecone Index** — nombre del índice (default: `rag-cvs`)

### 2. Subir los CVs
- Subir un archivo por persona en formato PDF, DOCX o TXT.
- **El nombre del archivo determina el nombre del agente.** Nombrar los archivos con el apellido o nombre de la persona. Ejemplos: `cardozo.pdf`, `didone.docx`, `jose_garcia.pdf`.
- Caracteres especiales y acentos se normalizan automáticamente (`josé.pdf` → agente `jose`).
- **El primer archivo cargado define el agente por defecto** (usado cuando la query no menciona a nadie).

### 3. Ingestar
Hacer clic en **🚀 Ingestar CVs**. El sistema:
1. Carga y fragmenta cada CV.
2. Genera embeddings con `text-embedding-3-small`.
3. Indexa cada CV en su propio namespace en Pinecone (`cv_<nombre>`).
4. Instancia un agente RAG independiente por persona.

### 4. Chatear
Escribir preguntas en lenguaje natural. Ejemplos:

```
¿Qué experiencia laboral tiene Cardozo?
```
> → Agente: **cardozo** (nombre detectado)

```
¿Cuántos años de experiencia tiene?
```
> → Agente: **primer CV cargado** (sin nombre → default)

```
Compará la experiencia de Cardozo y Didoné
```
> → Agentes: **cardozo + didoné** (múltiples nombres → respuestas combinadas)

Cada respuesta muestra un badge de routing (**DEFAULT** / **NAMED** / **MULTI**) indicando qué agentes se consultaron.

---

## Estructura del código

```
TP6_Rag_MultiAgent_Cardozo-Didone.py
│
├── validate_openai_key()        # Verifica la API key contra OpenAI
├── normalize_name()             # Sanitiza nombres de archivo a ASCII puro
├── load_document()              # Carga PDF / DOCX / TXT con LangChain loaders
├── build_agent()                # Construye un agente RAG por persona
│     ├── PineconeVectorStore    # Apunta al namespace de esa persona
│     ├── ChatPromptTemplate     # System prompt personalizado con el nombre
│     └── RunnableWithMessageHistory  # Memoria conversacional por sesión
├── ingest_single_cv()           # Fragmenta e indexa un CV en su namespace
├── detect_mentioned_persons()   # Detecta nombres propios en la query
├── orchestrate()                # Orquestador: rutea y combina respuestas
│     ├── Modo DEFAULT           # Sin nombre → primer agente
│     ├── Modo NAMED             # Un nombre → agente específico
│     └── Modo MULTI             # Varios nombres → todos los agentes relevantes
└── Streamlit UI                 # Sidebar de config + área de chat
```

---

## Notas técnicas

- El índice de Pinecone se crea automáticamente si no existe. Si ya existe, se reutiliza (no se borra entre ingestions).
- Cada namespace es `cv_<nombre_normalizado>` (solo ASCII, sin espacios).
- La memoria conversacional es independiente por agente y por sesión.
- La detección de personas es por substring case-insensitive; si se nombra `cardozo`, el sistema lo matchea contra el key `cardozo` del agente.
- La dimensión del índice es `1536`, correspondiente al modelo `text-embedding-3-small` de OpenAI.
