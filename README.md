#  MovieBot – Intelligent Movie Recommendation Chatbot

A smart chatbot that understands natural language queries and recommends movies using a hybrid approach of **filters + semantic search**.

---

##  Features

*  Filter-based search (actor, director, year, rating)
*  Semantic search using embeddings
*  Fuzzy matching for names (handles typos & partial inputs)
*  Conversational responses (ChatGPT-like style)
*  Fast and interactive UI (Streamlit)

---

## Project structure
```bash
├── app.py
├── chatbot.py
├── cleaned_movies.csv
├── moviebot.ipynb
├── README.md
├── requirements.txt
├── tmdb_5000_credits.csv
└── tmdb_5000_movies.csv
```

---

##  Setup Instructions

### 1️ Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️  Add Hugging Face Token (IMPORTANT)

Create a `.env` file in the root directory and add:

```env
HF_TOKEN=your_huggingface_token_here
```

warning: Without this, the chatbot will NOT work.

---

### 4️ Run the app

```bash
streamlit run app.py
```

---

## Example Queries

* "movies directed by christopher nolan"
* "comedy movies after 2015"
* "movies with leonardo dicaprio"
* "sci-fi movies with rating above 8"

---

##  How It Works

1.  Extract filters from user query
2.  Apply structured filtering
3.  Perform semantic similarity search
4.  Rank and return best results

---

##  Notes

* Do NOT hardcode your Hugging Face token
* Always use `.env` for security
* Make sure dataset preprocessing is done correctly

---



##  Future Improvements

* Better conversational memory
* Multi-turn dialogue handling
* Improved ranking logic
* Deployment (Streamlit Cloud / Hugging Face Spaces)

---

## Limitations

1.sometimes gives random outputs and the director filter seems to fail. 
2.cant capture the semantics properly when a non movie query is asked 
3.when given proper details the chatbot tries to capture all of the filters and it seems to not give proper result but gives the starting set of movies as the output.

---
