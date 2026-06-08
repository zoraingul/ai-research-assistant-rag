import os 
from dotenv import load_dotenv
import google.genai as genai

from vector_store import retrieve_chunks


#load api key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

#build prompt 
def build_prompt(chunks , question):
    context = '\n\n'.join(chunks)
    
    prompt = f"""
    You are an AI assistant. Use ONLY the context below to answer the question.
    Context:
    {context}
    Question:
    {question}

    If the answer is not in the context, say "I don't know based on the provided document."
    Answer clearly and concisely.
    """
    return prompt

#generate answer
def generate_answer(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# Main RAG Function
def ask_question(question, model_embed, index, chunks, k=3):
    # 1. Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(
        question,
        model_embed,
        index,
        chunks,
        k
    )

    # Build prompt
    prompt = build_prompt(retrieved_chunks, question)

    # Get LLM response
    answer = generate_answer(prompt)

    return answer