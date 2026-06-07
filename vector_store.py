from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def create_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    return embeddings

