import streamlit as st
from chatbot import chatbot, get_default_memory

st.set_page_config(
    page_title="MovieBot - Smart Movie Recommendations",
    page_icon="🎬",
    layout="wide",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .stChatMessage { padding: 12px; }
    .movie-header { font-size: 1.2em; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("🎬 MovieBot")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
    Ask me for movie recommendations! I understand:

    - **Genres**: *"comedy movies"*, *"sci-fi films"*
    - **Actors**: *"movies with Tom Hanks"*
    - **Directors**: *"directed by Nolan"*
    - **Years**: *"after 2015"*, *"from 2020"*
    - **Ratings**: *"rating above 8"*
    - **Combinations**: *"action movies after 2010 with rating above 7"*
    """
    )
    st.markdown("---")
    st.markdown("### Quick genres")
    cols = st.columns(3)
    genre_buttons = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Romance"]
    for i, genre in enumerate(genre_buttons):
        with cols[i % 3]:
            if st.button(genre, key=f"genre_{genre}"):
                genre_query = f"{genre.lower()} movies"
                if genre == "Sci-Fi":
                    genre_query = "science fiction movies"
                st.session_state["pending_query"] = genre_query

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory = get_default_memory()
        st.session_state.last_results = None
        st.rerun()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = get_default_memory()
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# --- Main Chat Area ---
st.title("🎬 Movie Recommendation Chatbot")
st.caption("Powered by semantic search & smart filtering")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle pending query from sidebar buttons
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask me for movie recommendations...")

# Use pending query if no direct input
if pending and not user_input:
    user_input = pending

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("🔍 Searching movies..."):
        response, updated_memory, results = chatbot(
            user_input,
            memory=st.session_state.memory,
            last_results=st.session_state.last_results,
        )

    # Update session state
    st.session_state.memory = updated_memory
    if results is not None:
        st.session_state.last_results = results

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
