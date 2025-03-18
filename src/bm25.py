from rank_bm25 import BM25Okapi


def search_bm(query,chunks,n):
    """Recherche les chunks les plus pertinents à partir d'une requête."""
    chunks_words = [doc.split(" ") for doc in chunks]
    
    bm25 = BM25Okapi(chunks_words)
    
    query_words = query.split(" ")
    scores = bm25.get_scores(query_words)
    sorted_scores = sorted(zip(scores, chunks), reverse=True)[:n]
    best_chunks = bm25.get_top_n(query_words, chunks, n)
    for i, (score, best_chunk) in enumerate(sorted_scores, start=1):
        print(f"Résultat {i}: (Score: {score:.4f})")
        print(f"➡️ {best_chunk}\n")

    return [chunk for _, chunk in sorted_scores]