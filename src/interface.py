import streamlit as st
from langchain_ollama import ChatOllama
import dotenv
import os
from src.search_bdd import search
from src.bm25 import search_bm
import chromadb
from sentence_transformers import SentenceTransformer

# Charger les variables d'environnement si nécessaire
dotenv.load_dotenv()

def run_interface(collection,embedding_model,chunks,NB_CONTEXT):
    st.title("Interface PROJET E4 Micropole")
    st.subheader("Saisissez votre requête ci-dessous.")

    # Initialisation du modèle ChatOllama
    llm = ChatOllama(
        model="deepseek-llm",
        device="gpu",  
        trust_remote_code=True,  # obligatoire pour les modèles HF
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )

    # Initialiser l'historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher les messages de l'historique de la conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Réagir à la saisie de l'utilisateur
    if prompt := st.chat_input("Écrivez votre question"):
        # Afficher le message de l'utilisateur dans le container de message
        st.chat_message("user").markdown(prompt)
        # Ajouter le message de l'utilisateur à l'historique des messages
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Effectuer la recherche dans la base de données
        search_semantic = search(prompt, collection, embedding_model, top_k=int(NB_CONTEXT))
        search_bm25 = search_bm(prompt, chunks, top_k=int(NB_CONTEXT))
        # Construire le context avec les résultats de la recherche
        context_semantic = "\n".join([result["document"] for result in search_semantic])
        print('sem')
        print(context_semantic)
        context_bm25 = "\n".join([result["document"] for result in search_bm25])
        print('bm25')
        print(context_bm25)
        
        # Vérification que le context_semantic n'est pas vide
        if not context_semantic or not context_bm25:
            st.write("Aucun contexte trouvé pour répondre à la question.")
            return
        
        context = context_semantic + "\n" + context_bm25

        # Préparer le prompt pour Deepseek
        deepseek_prompt = f"""
        Répond strictement à la question suivante en utilisant uniquement les informations présentes dans le contexte fourni. Aucune information externe ou spéculation n'est autorisée. Si le contexte ne permet pas de répondre de manière cohérente, indique clairement que la réponse ne peut être fournie : \n{context}

        Question : \n\n{prompt}
        """


        # Obtenir la réponse du modèle
        response = llm.invoke(deepseek_prompt).content.split('</think>')  # Assurez-vous que la chaîne 'think' existe
        print(response)
        # Afficher la réponse de l'assistant dans le container de message
        with st.chat_message("assistant"):
            st.markdown(response[0])
        # Ajouter la réponse de l'assistant à l'historique des messages
        st.session_state.messages.append({"role": "assistant", "content": response})