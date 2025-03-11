from langchain_community.vectorstores import Chroma
from document_split import *
import os
import shutil
from tqdm import tqdm

CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)

    # Utilisation de tqdm pour afficher la progression
    for chunk in tqdm(chunks, desc="Generating embeddings"):
        db.add_documents([chunk])

    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
