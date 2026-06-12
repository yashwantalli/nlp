# 🎬 MovieBot – Intelligent Movie Recommendation Chatbot

A smart chatbot that understands natural language queries and recommends movies using a hybrid approach of **filters + semantic search**.

---

## Features

* **Genre-based filtering** — action, comedy, sci-fi, thriller, etc.
* **Filter-based search** — actor, director, year, rating
* **Semantic search** using sentence embeddings (all-MiniLM-L6-v2)
* **Fuzzy matching** for names (handles typos & partial inputs)
* **Conversational memory** — follow-up queries remember context
* **Rich results** — shows year, rating, and genres for each movie
* **Quick genre buttons** in the sidebar for one-click search
* **Session-based state** — safe for multi-user Streamlit deployments

---

## Project Structure

```bash
├── app.py                  # Streamlit frontend (UI + session state)
├── chatbot.py              # Core logic (filters, search, response formatting)
├── cleaned_movies.csv      # Preprocessed movie dataset
├── moviebot.ipynb          # Exploratory notebook
├── README.md
├── requirements.txt
├── tmdb_5000_credits.csv   # Raw credits data
└── tmdb_5000_movies.csv    # Raw movies data
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yashwantalli/nlp.git
cd nlp
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Add Hugging Face Token (optional but recommended)

Create a `.env` file in the root directory and add:

```env
HF_TOKEN=your_huggingface_token_here
```

> Without this token, the chatbot falls back to regex-based filter extraction (which still works well for most queries).

---

### 4. Run the app

```bash
streamlit run app.py
```

---

## Example Queries

* `"comedy movies after 2015"`
* `"movies directed by Christopher Nolan"`
* `"movies with Leonardo DiCaprio"`
* `"sci-fi movies with rating above 8"`
* `"action movies from 2020"`
* `"thriller movies starring Tom Hanks"`
* `"tell me more about the first one"` (after getting results)

---

## How It Works

1. **Query classification** — determines if it's a movie-related query
2. **Filter extraction** — uses LLM (with regex fallback) to identify actor, director, genre, year, rating
3. **Structured filtering** — applies filters to the movie database
4. **Semantic search** — ranks remaining results by embedding similarity
5. **Response formatting** — presents results with ratings, years, and genres

---

## Architecture

```
User Query
    │
    ├─ Is movie-related? ──(no)──► "I specialize in movies..."
    │
    ├─ Extract filters (LLM + regex)
    │
    ├─ Has structured filters? ──(yes)──► Filter → Sort by rating
    │                           ──(no)───► Semantic search
    │
    ├─ Format results with details
    │
    └─ Return response + update memory
```

---

## Notes

* Do NOT hardcode your Hugging Face token
* Always use `.env` for security
* The chatbot works without an HF token (regex-only mode)
* Session memory resets when you start a new search topic

---

## Future Improvements

* Multi-turn dialogue with deeper context understanding
* User preference learning across sessions
* Improved ranking with collaborative filtering
* Deployment (Streamlit Cloud / Hugging Face Spaces)

---
