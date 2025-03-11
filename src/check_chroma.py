from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

CHROMA_PATH = os.getenv("CHROMA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# Charger la base
embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Vérifier le nombre de chunks
nb_chunks = db._collection.count()
print(f"✅ Chroma contient {nb_chunks} chunks/documents.")

# Optionnel : faire une recherche simple
query = "Qui est le directeur de l'innovation"  # Remplace par un mot-clé que tu penses présent
vector=embedding_function.embed_query(text=query)
results = db._similarity_search_with_relevance_scores(query, k=3)

print(f"\n🔍 Résultats pour la requête '{query}':")
if results:
    for i, res in enumerate(results):
        print(f"\n--- Résultat {i+1} ---\n{res[0].page_content}...")  # Affiche début du chunk
else:
    print("❌ Aucun résultat trouvé.")
