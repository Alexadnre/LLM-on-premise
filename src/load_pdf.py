from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(BDC_PATH):
    """
    Load all PDF documents from a specified directory.

    Args:
        BDC_PATH (str): The path to the directory containing PDF files.

    Returns:
        list: A list of documents loaded from the PDF files in the directory.
    """
    documents = []

    loader = PyPDFDirectoryLoader(BDC_PATH)  # Charge tous les PDF d'un dossier
    documents = loader.load()

    return documents













