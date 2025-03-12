import chromadb
from sentence_transformers import SentenceTransformer
import dotenv

EMBEDDING_MODEL=dotenv.get("EMBEDDING_MODEL")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path="db")  # Stockage persistant dans le dossier "db"
collection = chroma_client.get_or_create_collection(name="rag_bdc")  # Créer une collection

def index_chunks(chunks):
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
