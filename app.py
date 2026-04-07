import streamlit as st
from chatbot import chatbot

st.set_page_config(page_title="Movie Chatbot", page_icon="🎬")

st.title("🎬 Movie Recommendation Chatbot")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input
user_input = st.chat_input("Ask me for movies...")

if user_input:
    # show user
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # response
    with st.spinner("Thinking..."):
        response = chatbot(user_input)

    # show bot
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})