import os
from tqdm import tqdm

def index_chunks(chunks, collection, embedding_model):
    """Indexe les chunks de texte dans ChromaDB."""
    
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

