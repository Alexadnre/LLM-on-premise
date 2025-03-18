import os
from tqdm import tqdm

def index_chunks(chunks, collection, embedding_model):
    """
    Indexe les chunks de texte dans ChromaDB.

    Args:
        chunks (list): Liste des chunks de texte à indexer.
        collection (chromadb.Collection): Collection ChromaDB où les chunks seront ajoutés.
        embedding_model (SentenceTransformer): Modèle utilisé pour générer les embeddings.

    Returns:
        None
    """
    
    # Convertir les chunks en embeddings
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Ajouter chaque chunk à la base ChromaDB
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],  # ID unique pour chaque chunk
            documents=[chunk],  # Texte original
            embeddings=[embedding]  # Vecteur associé
        )

    print(f"✅ {len(chunks)} chunks indexés dans ChromaDB")

