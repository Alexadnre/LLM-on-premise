from src.load_pdf import load_documents
from src.chunking import split_text
from src.index_on_chroma import index_chunks
from src.search_bdd import search



texts = load_documents()
chunks = split_text([doc.page_content for doc in texts], chunk_size=800, chunk_overlap=80)
index_chunks(chunks)

search("QUels sont mes congés en 2025 ?", top_k=3)