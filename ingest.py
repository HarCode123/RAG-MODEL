import pymupdf
import numpy as np
import faiss
import pickle
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

DATA_PATH = "data/class 11 ncert"

def extract_text_from_pdf(pdf_path):
    text = ""

    doc = pymupdf.open(pdf_path)

    for page in doc:
        text += page.get_text()

    doc.close()
    return text



def load_all_pdfs():

    documents = []

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            path = os.path.join(DATA_PATH, file)

            text = extract_text_from_pdf(path)

            documents.append({
                "filename": file,
                "text": text
            })

            print(f"Loaded : {file}")

    return documents


if __name__ == "__main__":

    docs = load_all_pdfs()

    print("\nTotal PDFs :", len(docs))

    if len(docs) > 0:
        print("First file :", docs[0]["filename"])
        print("Characters :", len(docs[0]["text"]))
    else:
        print("No PDFs found. Check DATA_PATH.")
    chunks = []
    size = 1200

    for doc in docs:
        text = doc["text"]
        for i in range(0, len(text), size):
            chunks.append(text[i:i+size])

    print("Total Chunks :", len(chunks))

    # -------- Embeddings --------
    '''all_embeddings = []

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        while True:
            try:
                response = client.embeddings.create(
                    model="gemini-embedding-001",
                    input=batch
            )
                break
            except Exception as e:
                print("Rate limit hit. Waiting 40 seconds...")
                time.sleep(40)
        for item in response.data:
            all_embeddings.append(item.embedding)
        print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")'''
         
    #embeddings = np.array(all_embeddings, dtype="float32")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    # -------- FAISS --------
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("vectorstore", exist_ok=True)

    faiss.write_index(index, "vectorstore/index.faiss")

    with open("vectorstore/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index saved successfully!")