from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(BDC_PATH):
    
    documents = []

    loader = PyPDFDirectoryLoader(BDC_PATH)  # Charge tous les PDF d'un dossier
    documents = loader.load()

    return documents













