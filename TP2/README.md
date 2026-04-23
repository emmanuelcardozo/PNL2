# 🧠 CV RAG Chatbot

Aplicación de **Retrieval-Augmented Generation (RAG)** construida con Streamlit que permite consultar CVs en lenguaje natural. Cargás uno o dos currículums, se indexan automáticamente en una base de datos vectorial, y podés hacerle preguntas sobre los candidatos en un chat conversacional.

---

## ✨ Funcionalidades

- Carga de CVs en formato PDF, DOCX o TXT (hasta 2 archivos)
- Chunking e indexación automática en Pinecone
- Embeddings con OpenAI (`text-embedding-3-small`)
- LLM conversacional con `gpt-3.5-turbo` vía LangChain
- Historial de conversación por sesión
- Interfaz dark con diseño editorial

---

## 🏗️ Arquitectura

### Pipeline de Ingestión

```
CV (PDF/DOCX/TXT)
      ↓
Document Loader (LangChain)
      ↓
Text Splitter — chunk_size: 1000 / overlap: 100
      ↓
Embedding Model (text-embedding-3-small)
      ↓
Vector Database (Pinecone — dimensión 1536)
```

### Pipeline de Consulta

```
Pregunta del usuario
      ↓
Embed pregunta (text-embedding-3-small)
      ↓
Similarity Search en Pinecone (top-k: 4)
      ↓
Prompt Builder (pregunta + contexto + historial)
      ↓
LLM (gpt-3.5-turbo)
      ↓
Respuesta → Chat UI
```

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| Frontend | Streamlit |
| LLM | OpenAI `gpt-3.5-turbo` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Pinecone (Serverless, AWS us-east-1) |
| Orquestación | LangChain |
| Historial | `RunnableWithMessageHistory` |

---

## 📦 Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd cv-rag-chatbot

# Instalar dependencias
pip install streamlit langchain langchain-openai langchain-community langchain-pinecone pinecone-client pypdf docx2txt
```

---

## 🚀 Uso

```bash
streamlit run app5.py
```

En el sidebar de la app:

1. Ingresá tu **OpenAI API Key** (`sk-...`)
2. Ingresá tu **Pinecone API Key** (`pcsk-...`)
3. Configurá el nombre del índice y namespace (por defecto `rag-cvs` / `cvs`)
4. Subí entre 1 y 2 CVs en PDF, DOCX o TXT
5. Hacé click en **🚀 Ingestar CVs**
6. Una vez indexados, usá el chat para hacer preguntas sobre los candidatos

---

## 🔑 API Keys necesarias

| Servicio | Dónde obtenerla |
|---|---|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Pinecone | [app.pinecone.io](https://app.pinecone.io) |

> ⚠️ Las keys se ingresan en la interfaz y no se almacenan en ningún archivo. No las subas a un repositorio público.

---

## 📁 Estructura del proyecto

```
cv-rag-chatbot/
├── app5.py        # Aplicación principal
└── README.md      # Este archivo
```

---

## 💬 Ejemplos de preguntas

- *¿Cuántos años de experiencia tiene el candidato A?*
- *¿Qué tecnologías maneja el candidato B?*
- *¿Alguno de los candidatos tiene experiencia en gestión de equipos?*
- *Compará el perfil técnico de ambos candidatos.*

---

## 📄 Licencia

MIT
