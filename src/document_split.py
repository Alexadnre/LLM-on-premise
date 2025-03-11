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


    print("🎉 Traitement terminé !")
    return chunks


if __name__ == "__main__":
    splitted_docs = main()
    print(f"Nombre total de chunks : {len(splitted_docs)}")
