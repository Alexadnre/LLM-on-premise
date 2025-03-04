
import os
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings


DATA_PATH = "static/base_de_connaissance"
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

def embed(documents : list[Document]):
    embed_model = OllamaEmbeddings(model="deepseek-embed")  
    embeddings = embed_model.embed_documents(split_text(documents))
    return embeddings

def sanitize_filename(filename: str) -> str:
    """ Nettoie un nom de fichier en supprimant les espaces et caractères spéciaux. """
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

def main():
    documents = load_documents()
    splitted_documents = split_text(documents)
    embedded = embed(documents)
    save_chunks(splitted_documents, OUTPUT_PATH)
    return splitted_documents

if __name__ == "__main__":
    splitted_docs = main()
    print(f"Nombre total de chunks : {len(splitted_docs)}")

