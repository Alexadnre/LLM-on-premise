from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(BDC_PATH, COLLECTION_NAME):
    
    documents = []

    loader = PyPDFDirectoryLoader(COLLECTION_NAME)  # Charge tous les PDF d'un dossier
    documents = loader.load()

    return documents













