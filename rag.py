import os
import pickle
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Load FAISS index
index = faiss.read_index("vectorstore/index.faiss")

# Load text chunks
with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def ask_question(question):
    # Create embedding
    emb = client.embeddings.create(
        model="text-embedding-004",
        input=question
    )
    query_vector = emb.data[0].embedding
    query_vector = np.array([query_vector],dtype="float32")

    # Search top 3 chunks
    D, I = index.search(
        query_vector, 3
    )
     # Create context
    context = ""
    for i in I[0]:
        context += chunks[i] + "\n\n"

    # Ask Gemini
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the given NCERT context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content