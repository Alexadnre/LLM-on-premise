import os
import json
import numpy as np
from numpy.linalg import norm
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import ChatOllama

DATA_PATH = "static2/"
OUTPUT_PATH = "output"  

def load_documents():
    documents = []
    
    docx_loader = DirectoryLoader(DATA_PATH, glob="*.docx")
    documents.extend(docx_loader.load())

    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PDFPlumberLoader)
    documents.extend(pdf_loader.load())

    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents en {len(chunks)} chunks.")
    return chunks

def sanitize_filename(filename: str) -> str:
    """Nettoie un nom de fichier en supprimant les espaces et caractères spéciaux."""
    filename = os.path.basename(filename)
    filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return filename

def save_chunks(chunks: list[Document], output_path: str):
    os.makedirs(output_path, exist_ok=True)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        source_cleaned = sanitize_filename(source)

        file_path = os.path.join(output_path, f"{source_cleaned}_chunk_{i}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)
    
    print(f"Chunks sauvegardés dans {output_path}")

def get_embedding(text: str, llm: ChatOllama):
    """
    Demande au modèle deepseek-llm:latest de renvoyer l'embedding
    sous forme d'une liste de nombres en JSON.
    """
    prompt = (
        "Fournis-moi l'embedding pour le texte suivant sous forme d'une liste de nombres en JSON, "
        "sans explications ni texte supplémentaire.\n\n"
        f"{text}"
    )
    response = llm.invoke(prompt)
    try:
        # On suppose que le modèle retourne uniquement un JSON.
        embedding = json.loads(response.content)
        return embedding
    except json.JSONDecodeError as e:
        print("Erreur lors du décodage JSON pour le texte :", text[:50], "...\nErreur :", e)
        return None

def embed_documents_with_llm(documents: list[Document]):
    """
    Pour chaque chunk, on interroge deepseek-llm:latest afin d'obtenir un embedding.
    """
    # Initialisation de ChatOllama avec deepseek-llm:latest
    llm = ChatOllama(
        model="deepseek-llm:latest",
        device="cpu",            # ou "gpu" si approprié
        trust_remote_code=True,
        max_new_tokens=64,
        temperature=0.8,         # température basse pour plus de déterminisme
    )
    chunks = split_text(documents)
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding du chunk {i+1}/{len(chunks)}...")
        emb = get_embedding(chunk.page_content, llm)
        if emb is not None:
            embeddings.append(emb)
        else:
            embeddings.append([])  # Optionnel : on ajoute une liste vide en cas d'erreur
    return embeddings, chunks

def test_embeddings():
    """Teste que les embeddings ont bien été générés correctement."""
    documents = load_documents()
    embeddings, chunks = embed_documents_with_llm(documents)
    
    # 1. Vérifier que le nombre d'embeddings correspond au nombre de chunks
    if len(embeddings) != len(chunks):
        print(f"Erreur: {len(embeddings)} embeddings générés pour {len(chunks)} chunks.")
    else:
        print(f"[OK] {len(embeddings)} embeddings générés pour {len(chunks)} chunks.")
    
    # 2. Vérifier la dimension du premier embedding
    if embeddings and embeddings[0]:
        emb_dim = len(embeddings[0])
        print(f"Dimension du premier embedding : {emb_dim}")
    else:
        print("Aucun embedding n'a été généré pour le premier chunk.")
    
    # 3. Calculer la similarité cosinus entre le premier et le deuxième embedding (si disponibles)
    if len(embeddings) >= 2 and embeddings[0] and embeddings[1]:
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        cos_sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        print(f"Similarité cosinus entre le 1er et le 2ème embedding : {cos_sim:.3f}")
    else:
        print("Pas assez d'embeddings pour calculer la similarité.")

def main():
    documents = load_documents()
    embeddings, chunks = embed_documents_with_llm(documents)
    save_chunks(chunks, OUTPUT_PATH)
    print(f"Nombre total de chunks : {len(chunks)}")
    test_embeddings()

if __name__ == "__main__":
    main()
