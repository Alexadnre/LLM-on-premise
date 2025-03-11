from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util
import os

# Récupération et validation des variables d'environnement
CHROMA_PATH = os.getenv("CHROMA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not CHROMA_PATH:
    raise ValueError("⚠️ CHROMA_PATH n'est pas défini dans les variables d'environnement.")

if not EMBEDDING_MODEL:
    raise ValueError("⚠️ EMBEDDING_MODEL n'est pas défini dans les variables d'environnement.")

# Chargement du store Chroma
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL))

# Chargement du modèle SentenceTransformer une seule fois
sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search_and_rerank(query: str, top_k: int = 3):
    """Recherche dans ChromaDB et effectue un reranking basé sur la similarité cosinus."""
    
    print(f"\n🔎 Recherche pour : '{query}'")
    
    # Création du retriever Chroma
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Recherche initiale dans Chroma
    retrieved_chunks = retriever.get_relevant_documents(query)
    if not retrieved_chunks:
        print("⚠️ Aucun résultat trouvé.")
        return []
    
    print(f"✅ {len(retrieved_chunks)} documents récupérés.")

    # Encodage du texte
    query_embedding = sentence_transformer.encode(query, convert_to_tensor=True)
    chunk_embeddings = [sentence_transformer.encode(doc.page_content, convert_to_tensor=True) for doc in retrieved_chunks]

    # Calcul des similarités cosinus
    similarity_scores = [util.cos_sim(query_embedding, emb).item() for emb in chunk_embeddings]

    # Tri des résultats selon la similarité
    reranked_chunks = sorted(zip(retrieved_chunks, similarity_scores), key=lambda x: x[1], reverse=True)

    return reranked_chunks

# Exécution d'une requête de test
query = "MaIA "
results = search_and_rerank(query)

# Affichage des résultats classés
if results:
    print("\n📌 Résultats classés :\n")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"{i}. (Score: {score:.4f}) {doc.page_content}...")  # Tronquer pour lisibilité
