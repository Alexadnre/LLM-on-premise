import chromadb
from sentence_transformers import SentenceTransformer

def search(query, collection, embedding_model, top_k):
    """
    Recherche les chunks les plus pertinents à partir d'une requête.

    Args:
        query (str): Requête utilisateur.
        collection (chromadb.Collection): Collection ChromaDB où effectuer la recherche.
        embedding_model (SentenceTransformer): Modèle utilisé pour générer l'embedding de la requête.
        top_k (int): Nombre de résultats les plus pertinents à retourner.

    Returns:
        None
    """
    query_embedding = embedding_model.encode([query]).tolist()  # Convertir la requête en vecteur
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return [{"document": doc, "score": score} for doc, score in zip(results["documents"][0], results["distances"][0])]