from src.load_pdf import load_documents
from src.chunking import split_text
from src.index_on_chroma import index_chunks
from src.search_bdd import search



# texts = load_documents()
# chunks = split_text([doc.page_content for doc in texts], chunk_size=800, chunk_overlap=80)
# index_chunks(chunks)


print("Qui est le directeur de l'innovation ?")
search("Qui est le directeur de l'innovation ?", top_k=3)

print("Combien de salariés compte l'entreprise ?")
search("Combien de salariés compte l'entreprise ?", top_k=3)

print("Qui est maia ?")
search("Qui est maia ?", top_k=3)