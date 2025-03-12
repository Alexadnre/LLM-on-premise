import chromadb
import ollama
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# 📌 Variables globales
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm:l6-v2")  # Modèle d'embedding (par défaut DeepSeek)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "src\db")  # Dossier pour stocker la base de données Chroma
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bdc")  # Nom de la collection ChromaDB

# Initialisation de ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def get_embedding(text):
    """Génère un embedding à partir d'un texte en utilisant Ollama."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]  # Retourne directement le vecteur d'embedding

def search(query, top_k=3):
    """Recherche les chunks les plus pertinents à partir d'une requête."""
    
    query_embedding = get_embedding(query)  # Convertir la requête en vecteur
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    # Afficher les résultats
    for i, (doc, score) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"🔎 Résultat {i+1}: (Score: {score:.4f})\n{doc}\n{'-'*50}")
