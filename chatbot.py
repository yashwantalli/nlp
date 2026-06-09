import json
import os
import re

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    QUERY_SUFFIX,
    convert,
    fuzzy_match_name,
    get_director,
    make_filters,
)

load_dotenv()

movies = pd.read_csv("cleaned_movies.csv")

for col in ["genres", "keywords", "cast"]:
    movies[col] = movies[col].apply(convert)

movies["crew"] = movies["crew"].apply(get_director)

model = SentenceTransformer("all-MiniLM-L6-v2")

movie_embeddings = model.encode(movies["combined"].tolist(), show_progress_bar=True)

# ── filtering ────────────────────────────────────────────────────────────────


def filter_movies(df, filters):

    if filters["rating"] is not None:
        df = df[df["vote_average"] >= filters["rating"]]

    if filters["year"] is not None:
        df = df[df["year"] == filters["year"]]

    if filters.get("year_after") is not None:
        df = df[df["year"] >= filters["year_after"]]

    if filters.get("year_before") is not None:
        df = df[df["year"] < filters["year_before"]]

    if filters["actor"] is not None:
        actor_name = filters["actor"]
        df = df[df["cast"].apply(lambda names: fuzzy_match_name(actor_name, names))]

    if filters["director"] is not None:
        director_name = filters["director"]
        df = df[df["crew"].apply(lambda names: fuzzy_match_name(director_name, names))]
        df = df.sort_values(by=["vote_average", "popularity"], ascending=False)

    return df


# ── semantic search ──────────────────────────────────────────────────────────


def semantic_search(query, top_k=5, df=None):
    """Run semantic search, optionally restricted to a filtered *df*."""
    augmented = query + QUERY_SUFFIX

    query_embedding = model.encode([augmented])

    if df is not None:
        df = df.reset_index(drop=True)
        embeddings = movie_embeddings[df.index]
    else:
        df = movies
        embeddings = movie_embeddings

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_indices]


# ── hybrid search ────────────────────────────────────────────────────────────


def hybrid_search(movies_df, filters, query=None, top_k=5):

    filtered_df = filter_movies(movies_df, filters)

    if filtered_df.empty:
        relaxed_df = filter_movies(
            movies_df,
            make_filters(
                rating=filters.get("rating"),
                actor=filters.get("actor"),
                director=filters.get("director"),
            ),
        )

        if not relaxed_df.empty:
            return relaxed_df.head(min(top_k, len(relaxed_df)))

        print("No exact match, relaxing filters...")

        if query:
            return semantic_search(query, top_k)

        return movies_df.head(top_k)

    # normal flow
    if query:
        return semantic_search(query, top_k, df=filtered_df)

    return filtered_df.head(min(top_k, len(filtered_df)))


# ── query parsing ────────────────────────────────────────────────────────────


def extract_top_k(query):
    match = re.search(r"(\d+)\s+(movies|films)", query.lower())
    if match:
        return int(match.group(1))
    return 5


API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

token = os.getenv("HF_TOKEN", "")

HEADERS = {"Authorization": f"Bearer {token}"}


def query_llm(prompt):
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 100}},
        )

        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"]

        return ""

    except Exception:
        return ""


def extract_filters(query):

    filters = make_filters()

    query = query.lower()

    # rating
    rating_match = re.search(r"rating (above|greater than)?\s*(\d+(\.\d+)?)", query)
    if rating_match:
        filters["rating"] = float(rating_match.group(2))

    # year exact
    year_match = re.search(r"(19|20)\d{2}", query)
    if year_match:
        filters["year"] = int(year_match.group())

    # year after
    after_match = re.search(r"after\s+(19|20)\d{2}", query)
    if after_match:
        filters["year_after"] = int(after_match.group().split()[-1])

    # year before
    before_match = re.search(r"before\s+(19|20)\d{2}", query)
    if before_match:
        filters["year_before"] = int(before_match.group().split()[-1])

    # actor
    actor_match = re.search(
        r"(acted by|starring|with)\s+([a-zA-Z\s]+?)(?:\sin|\swith|\sdirected|$)", query
    )
    if actor_match:
        filters["actor"] = actor_match.group(2).strip()

    # director
    director_match = re.search(r"(directed by)\s+([a-zA-Z\s]+)", query)
    if director_match:
        filters["director"] = director_match.group(2).strip()

    return filters


def extract_filters_llm(query):

    prompt = f"""
Extract movie filters from this query and return ONLY JSON:

Query: "{query}"

Format:
{{
    "rating": null,
    "year": null,
    "year_after": null,
    "year_before": null,
    "actor": null,
    "director": null
}}
"""

    response = query_llm(prompt)

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        filters = json.loads(response[start:end])
        return filters

    except Exception:
        return extract_filters(query)


# ── domain classification ────────────────────────────────────────────────────

domain_text = "movie film actor director cinema story plot"
domain_embedding = model.encode([domain_text])
last_results = None


def is_movie_query(query):
    query_embedding = model.encode([query])
    score = cosine_similarity(query_embedding, domain_embedding)[0][0]
    return score > 0.2


def is_new_query(query):
    query = query.lower()
    new_query_keywords = [
        "movies",
        "movie",
        "films",
        "actor",
        "acted",
        "director",
        "rating",
        "genre",
    ]
    return any(word in query for word in new_query_keywords)


memory = make_filters()


def update_memory(old_filters, new_filters):

    if new_filters.get("actor") is not None:
        old_filters["actor"] = new_filters["actor"]
        old_filters["director"] = None

    elif new_filters.get("director") is not None:
        old_filters["director"] = new_filters["director"]
        old_filters["actor"] = None

    for key in ["rating", "year", "year_after", "year_before"]:
        if new_filters.get(key) is not None:
            old_filters[key] = new_filters[key]

    return old_filters


def reset_memory():
    global memory
    memory = make_filters()


def is_followup_query(query):

    if not any(memory.values()):
        return False

    query_embedding = model.encode([query])

    memory_text = " ".join([str(v) for v in memory.values() if v])

    if not memory_text:
        return False

    memory_embedding = model.encode([memory_text])

    score = cosine_similarity(query_embedding, memory_embedding)[0][0]

    return score > 0.25


# ── main chatbot entry point ─────────────────────────────────────────────────


def chatbot(query):

    global memory
    global last_results

    if "explain" in query:
        if last_results is not None and len(last_results) > 0:
            if "first" in query:
                idx = 0
            elif "second" in query:
                idx = 1
            elif "third" in query:
                idx = 2
            else:
                idx = 0  # default

            if idx < len(last_results):
                movie = last_results.iloc[idx]

                if "overview" in movie:
                    return f"{movie['title']}:\n{movie['overview']}"
                else:
                    return f"{movie['title']}:\nNo description available."

        return "I don't have a previous movie list to explain."

    if not is_movie_query(query) and not is_followup_query(query):
        return ["Sorry, I can only help with movie-related queries."]

    if is_new_query(query):
        memory = make_filters()

    filters = extract_filters_llm(query)
    if not any(filters.values()):
        filters = extract_filters(query)
    if filters["actor"]:
        filters["actor"] = filters["actor"].strip().lower()

    if filters["director"]:
        filters["director"] = filters["director"].strip().lower()
    memory = update_memory(memory, filters)
    top_k = extract_top_k(query)

    if any([filters["rating"], filters["year"], filters["actor"], filters["director"]]):
        results = hybrid_search(movies, memory, None, top_k)
    else:
        results = hybrid_search(movies, memory, query, top_k)
    if isinstance(results, list):
        return results

    if isinstance(results, str):
        return results

    if "title" in results:
        titles = results["title"].tolist()

    if not titles:
        return ["No movies found for your query."]

    # dynamic message
    if filters["actor"]:
        intro = f"Here are movies featuring {filters['actor'].title()}:"
    elif filters["director"]:
        intro = f"Here are movies directed by {filters['director'].title()}:"
    elif filters["rating"] or filters["year"]:
        intro = "Here are movies matching your filters:"
    else:
        intro = "Here are some movies you might like:"

    formatted = [intro]

    for i, movie in enumerate(titles, 1):
        formatted.append(f"{i}. {movie}")
    last_results = results

    return "\n".join(formatted)
