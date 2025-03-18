from src.load_pdf import load_documents
from src.chunking import split_text
from src.index_on_chroma import index_chunks
from src.search_bdd import search

import chromadb
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# 📌 Variables globales
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")  # Modèle d'embedding (par défaut DeepSeek)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")  # Dossier pour stocker la base de données Chroma
COLLECTION_NAME = os.getenv("COLLECTION_NAME")  # Nom de la collection ChromaDB
BDC_PATH = os.getenv("BDC_PATH")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)  
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)  # Créer une collection






texts = load_documents(BDC_PATH)
chunks = split_text([doc.page_content for doc in texts], chunk_size=800, chunk_overlap=80)
index_chunks(chunks, collection, embedding_model)



search("Qui est le directeru de l'innovation",collection=collection,embedding_model=embedding_model,top_k=3)