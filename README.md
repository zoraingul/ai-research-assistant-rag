# Document Intelligence Platform

A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and interact with them through natural language questions.

The system combines semantic retrieval using FAISS and Sentence Transformers with Gemini 2.5 Flash to provide context-aware answers grounded in document content.

---

## Features

* PDF document ingestion
* Automatic text extraction and chunking
* Semantic embeddings using Sentence Transformers
* Vector similarity search using FAISS
* Context-aware question answering with Gemini
* Interactive Streamlit chat interface
* Modular architecture for easy extension

---

## System Architecture

```text
PDF Upload
    │
    ▼
Text Extraction
    │
    ▼
Chunking
    │
    ▼
Embedding Generation
    │
    ▼
FAISS Vector Index
    │
    ▼
User Question
    │
    ▼
Question Embedding
    │
    ▼
Similarity Search
    │
    ▼
Relevant Chunks Retrieved
    │
    ▼
Gemini 2.5 Flash
    │
    ▼
Generated Answer
```

---

## Tech Stack

* Python
* Streamlit
* Sentence Transformers
* FAISS
* Gemini 2.5 Flash
* NumPy
* PyPDF2

---

## Installation

```bash
git clone <repository-url>

cd document-intelligence-platform

python -m venv env

env\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run streamlit_app.py
```

---

## Example Use Cases

* Research paper analysis
* Academic document exploration
* Technical documentation search
* Knowledge base querying
* Report summarization

---

## Future Improvements

* Multi-document retrieval
* Source citations and page references
* Hybrid search (keyword + semantic)
* Conversation memory
* Deployment to cloud platform

```
```
