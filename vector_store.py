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

def build_faiss_index(embddings):
    dimensions = embddings.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(embddings)
    return index

def retrieve_chunks(question , model , index , chunks , k=3):
    question_embedding = model.encode(question)
    question_embedding = np.array(question_embedding).astype("float32")
    question_embedding = question_embedding.reshape(1 , -1)
    
    distances , indeces = index.search (
        question_embedding , 
        k
    )
    
    retrieved_chunks = []
    
    for idx in indeces[0]:
        retrieved_chunks.append(chunks[idx])
    
    return retrieved_chunks