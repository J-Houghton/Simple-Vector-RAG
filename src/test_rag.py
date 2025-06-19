import openai
import os
import argparse
# NumPy is no longer needed in the search function, but still good for the future
import numpy as np 
from dotenv import load_dotenv
from db import MilvusClient

# --- Configuration ---
load_dotenv()

# OpenAI Configuration
openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4"

# Oracle Database Configuration
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_DSN = os.environ.get("DB_DSN")

def embed_query(text: str) -> list[float]:
    """Generates a vector embedding for a given text query using OpenAI."""
    print("Embedding user query...")
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def search_database(query_vector: list[float], top_k: int = 5) -> list[tuple]:
    """Searches the Milvus database for the most relevant chunks."""
    print(f"Searching for top {top_k} relevant chunks in Milvus...")
    try:
        milvus = MilvusClient()
        results = milvus.search_vectors(query_vector, top_k=top_k)
        if results is None:
            print("Milvus search failed.")
            return []
        print("Found relevant chunks.")
        # Return as list of tuples (chunk_text, source_document, distance)
        return [(r["chunk_text"], r["source_document"], r["distance"]) for r in results]
    except Exception as e:
        print(f"Database search failed: {e}")
        return []

def generate_answer(question: str, context_chunks: list[tuple]):
    """Generates an answer using an LLM based on the retrieved context."""
    if not context_chunks:
        print("No context found, cannot generate an answer.")
        return

    print("Sending request to LLM for final answer generation...")
    context = "\n---\n".join([row[0] for row in context_chunks])
    
    prompt = f"""
    You are a helpful assistant for a technical community. Answer the user's question based only on the following context.
    If the context does not contain the answer, state that you don't have enough information from the provided documents.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    response = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Test the RAG pipeline with a user question.")
    parser.add_argument("question", type=str, help="The question to ask the system.")
    args = parser.parse_args()
    user_question = args.question

    print(f"\nProcessing question: '{user_question}'")
    
    query_vector = embed_query(user_question)
    retrieved_chunks = search_database(query_vector)

    if not retrieved_chunks:
        print("Could not find any relevant information in the knowledge base.")
        return

    print("\n--- RETRIEVED CONTEXT ---\n")
    for i, (text, source, distance) in enumerate(retrieved_chunks):
        print(f"Chunk {i+1} (Source: {source}, Distance: {distance:.4f}):\n{text}\n")
    
    final_answer = generate_answer(user_question, retrieved_chunks)

    print("\n--- FINAL ANSWER ---\n")
    print(final_answer)

if __name__ == "__main__":
    main()