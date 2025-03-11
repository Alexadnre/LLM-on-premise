# Charger la base de données existante
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os


CHROMA_PATH = os.getenv("CHROMA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)

# Récupérer tous les documents stockés
data = db.get()

# Afficher les clés disponibles
print(data.keys())

# Afficher un aperçu des données
for i in range(min(3, len(data["ids"]))):  # Afficher les 3 premiers éléments
    print(f"ID: {data['ids'][i]}")
    print(f"Document: {data['documents'][i]}")
    print(f"Embedding: {data['embeddings'][i][:5]}...")  # Affichage partiel
    print("-" * 50)
