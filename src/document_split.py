import os
import numpy as np
import pdfplumber
from numpy.linalg import norm
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from docx2pdf import convert
from tqdm import tqdm  # Ajout de tqdm pour la barre de progression
import re
from dotenv import load_dotenv


# Charger les variables d'environnement
load_dotenv()

# Utilisation des variables d'environnement
DATA_PATH = os.getenv("DATA_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


# def convert_all_docx_in_folder(folder_path):
#     # Parcours tous les fichiers du dossier
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".docx"):
#             input_path = os.path.join(folder_path, filename)
#             output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.pdf")
#             # Conversion du fichier .docx en .pdf
#             convert(input_path, output_path)
#             print(f"Le fichier {filename} a été converti en PDF.")
#             os.remove(input_path)



def load_documents():
    documents = []
    
    # convert_all_docx_in_folder(DATA_PATH)

    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PDFPlumberLoader)
    documents.extend(pdf_loader.load())
    print(f"📄 Chargé {len(documents)} pages de documents PDF.")
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


def embed(chunks: list[Document]):
    try:
        embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL)  
    except Exception as e:
        print(f"❌ Erreur avec Ollama : {e}")
        return [], []

    texts = [chunk.page_content for chunk in chunks]

    embeddings = []
    print("🔄 Encodage des documents...")
    
    for text in tqdm(texts, desc="📄 Encodage en cours", unit="chunk"):
        try:
            embeddings.append(embed_model.embed_documents([text])[0])
        except Exception as e:
            print(f"⚠️ Erreur d'encodage pour un chunk : {e}")

    return embeddings


def sanitize_filename(filename: str) -> str:
    """ Nettoie un nom de fichier en supprimant les espaces et caractères spéciaux. """
    filename = os.path.basename(filename)
    filename = re.sub(r"[^\w\-.]", "_", filename)  # Garde lettres, chiffres, tirets et points
    return filename


def save_chunks(chunks: list[Document], output_path: str):
    chunk_path = os.path.join(output_path, 'chunks')
    os.makedirs(chunk_path, exist_ok=True)  # Création du sous-dossier 'chunks'

    for i, chunk in tqdm(enumerate(chunks), desc="💾 Sauvegarde des chunks", total=len(chunks), unit="chunk"):
        source = chunk.metadata.get('source', 'unknown')
        source_cleaned = sanitize_filename(source)

        file_path = os.path.join(chunk_path, f"{source_cleaned}_chunk_{i}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)
    
    print(f"✅ Chunks sauvegardés dans {chunk_path}")


def save_embeddings(embeddings: list, output_path: str):
    embedding_path = os.path.join(output_path, 'embedding')
    os.makedirs(embedding_path, exist_ok=True)  # Création du sous-dossier 'embedding'

    for i, emb in tqdm(enumerate(embeddings), desc="💾 Sauvegarde des embeddings", total=len(embeddings), unit="embedding"):
        file_path = os.path.join(embedding_path, f"embedding_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(emb))
    print(f"✅ Embeddings sauvegardés dans {embedding_path}")



def main():
    print("🔄 Chargement des documents...")
    documents = load_documents()
    
    if not documents:
        print("❌ Aucun document trouvé. Fin du script.")
        return
    
    print("🔄 Division des documents en chunks...")
    chunks = split_text(documents)

    if not chunks:
        print("❌ Échec de la division des documents.")
        return

    print("🔄 Génération des embeddings...")
    embeddings = embed(chunks)

    if not embeddings:
        print("❌ Échec de la génération des embeddings.")
        return

    print("💾 Sauvegarde des chunks...")
    save_chunks(chunks, OUTPUT_PATH)

    print("💾 Sauvegarde des embeddings...")
    save_embeddings(embeddings, OUTPUT_PATH)

    print(f"✅ Nombre total de chunks : {len(chunks)}")


    print("🎉 Traitement terminé !")
    return chunks


if __name__ == "__main__":
    splitted_docs = main()
    print(f"Nombre total de chunks : {len(splitted_docs)}")
