from pdf_processor import extract_text_from_pdf, chunk_text
from vector_store import load_embedding_model, create_embeddings, build_faiss_index
from rag_pipeline import ask_question

text = extract_text_from_pdf("sample.pdf")
chunks = chunk_text(text)

model = load_embedding_model()
embeddings = create_embeddings(chunks, model)
index = build_faiss_index(embeddings)

answer = ask_question(
    "What is AI?",
    model,
    index,
    chunks
)

print(answer)