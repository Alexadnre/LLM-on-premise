from rank_bm25 import BM25Okapi

def search_bm(query, chunks, top_k):
    """Recherche les chunks les plus pertinents à partir d'une requête et retourne une liste de JSON."""
    chunks_words = [doc.split(" ") for doc in chunks]
    bm25 = BM25Okapi(chunks_words)
    query_words = query.split(" ")
    scores = bm25.get_scores(query_words)
    sorted_scores = sorted(zip(scores, chunks), reverse=True)[:top_k]
    result = [{"document": doc, "score": score} for score, doc in sorted_scores]
    
    return result
