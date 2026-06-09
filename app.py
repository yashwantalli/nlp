import logging
import streamlit as st
from chatbot import chatbot

logger = logging.getLogger(__name__)

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
        try:
            response = chatbot(user_input)
        except Exception as e:
            logger.error("Chatbot error: %s", e, exc_info=True)
            response = "Something went wrong while processing your request. Please try again."

    # show bot
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})