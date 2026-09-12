from ingest import extract_text_from_pdf
import os
def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    folder = "data/class 11 ncert"
    all_text = ""

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            all_text += extract_text_from_pdf(path) + "\n"

    chunks = chunk_text(all_text)

    print("Total characters:", len(all_text))
    print("Total chunks:", len(chunks))
    print("\nFirst chunk:\n")
    print(chunks[0][:500])