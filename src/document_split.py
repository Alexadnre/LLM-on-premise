# Installation a faire au préalable : 
# pip install langchain-community langchain langchain_openai
# pip install unstructured python-docx

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document



data_path = "static/base_de_connaisance"

def load_documents():
    loader = DirectoryLoader(data_path, glob="*.docx")
    documents = loader.load()
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

# faire un main qui renvoie le document splitté 


