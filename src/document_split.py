import os
import numpy as np
from numpy.linalg import norm
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings


DATA_PATH = "static2/"
OUTPUT_PATH = "output2"  

def load_documents():
    documents = []
    
    docx_loader = DirectoryLoader(DATA_PATH, glob="*.docx")
    documents.extend(docx_loader.load())

    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PDFPlumberLoader)
    documents.extend(pdf_loader.load())

    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents en {len(chunks)} chunks.")
    return chunks

def embed(documents: list[Document]):
    embed_model = OllamaEmbeddings(model="deepseek-llm:latest")  
    chunks = split_text(documents)
    texts = [chunk.page_content for chunk in chunks]
    test_text = ["Bonjour, ceci est un test."]
    embeddings = embed_model.embed_documents(test_text)
    print(embeddings)

    return embeddings, chunks

def sanitize_filename(filename: str) -> str:
    """ Nettoie un nom de fichier en supprimant les espaces et caractères spéciaux. """
    filename = os.path.basename(filename)
    filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return filename

def save_chunks(chunks: list[Document], output_path: str):
    os.makedirs(output_path, exist_ok=True)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        source_cleaned = sanitize_filename(source)

        file_path = os.path.join(output_path, f"{source_cleaned}_chunk_{i}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)
    
    print(f"Chunks sauvegardés dans {output_path}")

def save_embeddings(embeddings: list, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    for i, emb in enumerate(embeddings):
        file_path = os.path.join(output_path, f"embedding_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(emb))
    print(f"Embeddings sauvegardés dans {output_path}")

def test_embeddings():
    """Teste que les embeddings ont bien été générés correctement."""
    documents = load_documents()
    embeddings, chunks = embed(documents)

    if len(embeddings) >= 2:
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        cos_sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        print(f"Similarité cosinus entre le 1er et le 2ème embedding : {cos_sim:.3f}")
    else:
        print("Pas assez d'embeddings pour calculer la similarité.")

def main():
    documents = load_documents()
    embeddings, chunks = embed(documents)
    save_chunks(chunks, OUTPUT_PATH)
    save_embeddings(embeddings, OUTPUT_PATH)
    print(f"Nombre total de chunks : {len(chunks)}")
    # Lancement du test d'embeddings
    test_embeddings()
    return chunks

if __name__ == "__main__":
    splitted_docs = main()
    print(f"Nombre total de chunks : {len(splitted_docs)}")

