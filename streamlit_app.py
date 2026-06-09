import streamlit as st

from pdf_processor import extract_text_from_pdf, chunk_text
from vector_store import (
    load_embedding_model,
    create_embeddings,
    build_faiss_index
)
from rag_pipeline import ask_question



st.set_page_config(
    page_title="Document Intelligence Platform",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 10px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "model" not in st.session_state:
    st.session_state.model = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.title("Document Intelligence")

    st.caption(
        "Semantic document search and question answering "
        "using Retrieval-Augmented Generation."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"]
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Document Status")

    if st.session_state.ready:

        st.success("Document Loaded")

        st.write(
            f"**File:** {st.session_state.document_name}"
        )

        st.write(
            f"**Chunks:** {len(st.session_state.chunks)}"
        )

        st.write(
            "**Embedding Model:** all-MiniLM-L6-v2"
        )

        st.write(
            "**Vector Index:** FAISS"
        )

        st.write(
            "**LLM:** Gemini 2.5 Flash"
        )

    else:
        st.info("No document loaded")


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Document Intelligence Platform")

st.caption(
    "Upload a PDF and interact with it through semantic search, "
    "vector retrieval, and contextual question answering."
)

st.divider()


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------
if uploaded_file is not None and not st.session_state.ready:

    with st.spinner("Processing document..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        text = extract_text_from_pdf("temp.pdf")

        chunks = chunk_text(text)

        model = load_embedding_model()

        embeddings = create_embeddings(
            chunks,
            model
        )

        index = build_faiss_index(
            embeddings
        )

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.model = model
        st.session_state.document_name = uploaded_file.name
        st.session_state.ready = True

    st.success("Document processed successfully.")
    st.rerun()


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------
if st.session_state.ready:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Document Chunks",
            len(st.session_state.chunks)
        )

    with col2:
        st.metric(
            "Vector Index",
            "FAISS"
        )

    with col3:
        st.metric(
            "Language Model",
            "Gemini"
        )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
if st.session_state.ready:

    question = st.chat_input(
        "Ask a question about the uploaded document"
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Generating response..."):

                answer = ask_question(
                    question,
                    st.session_state.model,
                    st.session_state.index,
                    st.session_state.chunks
                )

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

else:

    st.info(
        "Upload a PDF document to begin semantic search and question answering."
    )

