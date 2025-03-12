from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

BDC_PATH = os.getenv("BDC_PATH")

def load_documents():
    
    documents = []

    loader = PyPDFDirectoryLoader(BDC_PATH)  # Charge tous les PDF d'un dossier
    documents = loader.load()

    return documents







