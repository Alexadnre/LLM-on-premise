import re

def split_text(text_list, chunk_size=500, chunk_overlap=50):

    all_chunks = []  # Liste pour stocker tous les chunks

    for text in text_list:  # Parcourir chaque texte/document
        sentences = re.split(r'(?<=[.!?])\s+', text)  # Découpe en phrases

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else: 
                chunks.append(current_chunk.strip())  # Stocke le chunk
                current_chunk = sentence + " "  # Démarre un nouveau chunk
        
        if current_chunk:  # Ajoute le dernier chunk restant
            chunks.append(current_chunk.strip())

        # Appliquer le chevauchement si demandé
        if chunk_overlap > 0:
            overlapped_chunks = []
            for i in range(len(chunks)):
                overlap_text = " ".join(chunks[max(0, i-1)].split()[-chunk_overlap:]) if i > 0 else ""
                overlapped_chunks.append(overlap_text + " " + chunks[i])
            all_chunks.extend(overlapped_chunks)  # Ajouter à la liste globale
        else:
            all_chunks.extend(chunks)

    return all_chunks 