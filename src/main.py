import streamlit as st
from langchain_ollama import ChatOllama

st.title("🦜🔗 Langchain Quickstart App")

# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
# https://python.langchain.com/docs/integrations/chat/ollama/
# https://streamlit.io/generative-ai

llm = ChatOllama(
        model="deepseek-r1:8b",
        device="gpu",
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Write your input"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Deepseek: {llm.invoke(prompt).content.split('</think>')[1]}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})