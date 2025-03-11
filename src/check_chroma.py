from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

CHROMA_PATH = os.getenv("CHROMA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# Charger la base
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)

# Vérifier le nombre de chunks
nb_chunks = db._collection.count()
print(f"✅ Chroma contient {nb_chunks} chunks/documents.")

# Optionnel : faire une recherche simple
query = "test"  # Remplace par un mot-clé que tu penses présent
results = db.similarity_search(query, k=3)

print(f"\n🔍 Résultats pour la requête '{query}':")
if results:
    for i, res in enumerate(results):
        print(f"\n--- Résultat {i+1} ---\n{res.page_content[:300]}...")  # Affiche début du chunk
else:
    print("❌ Aucun résultat trouvé.")
