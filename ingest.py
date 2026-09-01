import pymupdf
import os

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