# Installation a faire au préalable : 
# pip install langchain-community langchain langchain_openai
# pip install unstructured python-docx pypdf

from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import nltk

data_path = "static/base_de_connaissance"

def load_documents():
    doc_loader = DirectoryLoader(data_path, glob="*.docx")
    doc_documents = doc_loader.load()

    pdf_loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    documents = doc_documents + pdf_documents
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

documents = load_documents()

splitted_documents = split_text(documents)


##### A AMELIORER : 

# Pouvoir traiter différent type de doc tel que un pdf ou autre 

# -> Normalement c'est fait pour le pdf (fin chez moi ça marche)

# faire un main qui renvoie le document splitté  -> pas vraiment utile on est en python


