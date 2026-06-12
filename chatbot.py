import pandas as pd
import ast
import re
import json
import os
import numpy as np
import requests
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# --- Data Loading & Preprocessing ---

movies = pd.read_csv("cleaned_movies.csv")

VALID_GENRES = [
    "action", "adventure", "animation", "comedy", "crime", "documentary",
    "drama", "family", "fantasy", "history", "horror", "music", "mystery",
    "romance", "science fiction", "thriller", "tv movie", "war", "western"
]


def convert(text):
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return [i["name"] for i in data]
        elif isinstance(data, list):
            return data
    except (ValueError, SyntaxError):
        pass
    return []


def get_director(text):
    try:
        if isinstance(text, list):
            data = text
        else:
            data = ast.literal_eval(text)
        return [
            i["name"]
            for i in data
            if isinstance(i, dict) and i.get("job") == "Director"
        ]
    except (ValueError, SyntaxError):
        return []


for col in ["genres", "keywords", "cast"]:
    movies[col] = movies[col].apply(convert)

movies["crew"] = movies["crew"].apply(get_director)

# Lowercase genre lists for matching
movies["genres_lower"] = movies["genres"].apply(
    lambda x: [g.lower() for g in x] if isinstance(x, list) else []
)

# --- Embedding Model ---

model = SentenceTransformer("all-MiniLM-L6-v2")
movie_embeddings = model.encode(movies["combined"].tolist(), show_progress_bar=True)

# Precompute domain embedding for movie-query detection
DOMAIN_TEXT = "movie film actor director cinema story plot genre recommend suggestion"
domain_embedding = model.encode([DOMAIN_TEXT])

# --- HuggingFace LLM ---

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def query_llm(prompt):
    if not HF_TOKEN:
        return ""
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 150}},
            timeout=10,
        )
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return ""
    except (requests.RequestException, KeyError, IndexError):
        return ""


# --- Filter Extraction ---


def extract_filters(query):
    """Extract filters from query using regex patterns."""
    filters = {
        "rating": None,
        "year": None,
        "year_after": None,
        "year_before": None,
        "actor": None,
        "director": None,
        "genre": None,
    }

    q = query.lower()

    # Rating: "rating above 7", "rated over 8.5", "rating > 7"
    rating_match = re.search(
        r"rat(?:ing|ed)\s*(?:above|over|greater than|>|more than)?\s*(\d+\.?\d*)", q
    )
    if rating_match:
        filters["rating"] = float(rating_match.group(1))

    # Year exact: "from 2015", "in 2020", "of 2019", or standalone year
    year_exact = re.search(r"(?:from|in|of|year)\s+((?:19|20)\d{2})\b", q)
    if year_exact:
        filters["year"] = int(year_exact.group(1))

    # Year after: "after 2015", "since 2010", "post 2000"
    after_match = re.search(r"(?:after|since|post)\s+((?:19|20)\d{2})", q)
    if after_match:
        filters["year_after"] = int(after_match.group(1))
        filters["year"] = None  # after takes priority over exact

    # Year before: "before 2010", "pre 2005"
    before_match = re.search(r"(?:before|pre|until)\s+((?:19|20)\d{2})", q)
    if before_match:
        filters["year_before"] = int(before_match.group(1))
        filters["year"] = None

    # Director: "directed by X", "by director X", "X directed"
    director_match = re.search(
        r"(?:directed by|director)\s+([a-zA-Z\s.]+?)(?:\s+(?:with|in|after|before|rating|from|starring|acted)|$)",
        q,
    )
    if not director_match:
        director_match = re.search(
            r"(?:by)\s+([a-zA-Z\s.]+?)(?:\s+(?:movies?|films?)|$)", q
        )
    if director_match:
        name = director_match.group(1).strip()
        if len(name.split()) <= 4 and len(name) > 2:
            filters["director"] = name

    # Actor: "starring X", "with X", "acted by X", "movies of X"
    actor_match = re.search(
        r"(?:starring|acted by|featuring|with actor)\s+([a-zA-Z\s.]+?)(?:\s+(?:directed|in|after|before|rating|from)|$)",
        q,
    )
    if not actor_match:
        actor_match = re.search(
            r"(?:with)\s+([a-zA-Z\s.]+?)(?:\s+(?:directed|in|after|before|rating|from|and)|$)",
            q,
        )
    if actor_match and not filters["director"]:
        name = actor_match.group(1).strip()
        if len(name.split()) <= 4 and len(name) > 2:
            filters["actor"] = name

    # Genre: match against known genres
    for genre in VALID_GENRES:
        if genre in q or genre.replace(" ", "-") in q:
            filters["genre"] = genre
            break

    # Also match "sci-fi" -> "science fiction", "rom-com" -> romance + comedy
    if "sci-fi" in q or "scifi" in q:
        filters["genre"] = "science fiction"
    elif "rom-com" in q or "romcom" in q:
        filters["genre"] = "romance"

    return filters


def extract_filters_llm(query):
    """Try to extract filters with LLM, fall back to regex."""
    prompt = f"""Extract movie filters from this query. Return ONLY valid JSON.

Query: "{query}"

Return format:
{{"rating": null, "year": null, "year_after": null, "year_before": null, "actor": null, "director": null, "genre": null}}

Rules:
- genre must be one of: action, adventure, animation, comedy, crime, drama, family, fantasy, history, horror, music, mystery, romance, science fiction, thriller, war, western
- Use null for any filter not mentioned
- actor/director should be proper names only
"""
    response = query_llm(prompt)
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            filters = json.loads(response[start:end])
            # Validate genre
            if filters.get("genre") and filters["genre"].lower() not in VALID_GENRES:
                filters["genre"] = None
            return filters
    except (json.JSONDecodeError, ValueError):
        pass
    return extract_filters(query)


# --- Filtering ---


def filter_movies(df, filters):
    """Apply structured filters to movie dataframe."""
    result = df.copy()

    if filters.get("rating") is not None:
        result = result[result["vote_average"] >= filters["rating"]]

    if filters.get("year") is not None:
        result = result[result["year"] == filters["year"]]

    if filters.get("year_after") is not None:
        result = result[result["year"] >= filters["year_after"]]

    if filters.get("year_before") is not None:
        result = result[result["year"] < filters["year_before"]]

    if filters.get("genre") is not None:
        genre = filters["genre"].lower()
        result = result[
            result["genres_lower"].apply(lambda g: genre in g if isinstance(g, list) else False)
        ]

    if filters.get("actor") is not None:
        actor_name = filters["actor"].lower().strip()

        def match_actor(cast_list):
            if not isinstance(cast_list, list):
                return False
            for name in cast_list:
                name_lower = name.lower().strip()
                if actor_name == name_lower:
                    return True
                score = fuzz.partial_ratio(actor_name, name_lower)
                threshold = 90 if len(actor_name) <= 4 else 75
                if score > threshold:
                    return True
            return False

        result = result[result["cast"].apply(match_actor)]

    if filters.get("director") is not None:
        director_name = filters["director"].lower().strip()

        def match_director(crew_list):
            if not isinstance(crew_list, list):
                return False
            for name in crew_list:
                name_lower = name.lower().strip()
                if director_name == name_lower:
                    return True
                score = fuzz.partial_ratio(director_name, name_lower)
                threshold = 90 if len(director_name) <= 4 else 75
                if score > threshold:
                    return True
            return False

        result = result[result["crew"].apply(match_director)]

    # Always sort by rating + popularity
    result = result.sort_values(by=["vote_average", "popularity"], ascending=False)
    return result


# --- Semantic Search ---


def semantic_search(query, top_k=5):
    """Search movies by semantic similarity to query."""
    enriched = query + " movie plot story theme character"
    query_embedding = model.encode([enriched])
    similarities = cosine_similarity(query_embedding, movie_embeddings)[0]
    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return movies.iloc[top_indices]


def semantic_search_filtered(df, query, top_k=5):
    """Semantic search within a pre-filtered set of movies."""
    if df.empty:
        return df

    # Use original indices to correctly index into movie_embeddings
    original_indices = df.index.tolist()
    enriched = query + " movie plot story theme character"
    query_embedding = model.encode([enriched])
    subset_embeddings = movie_embeddings[original_indices]
    similarities = cosine_similarity(query_embedding, subset_embeddings)[0]
    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_indices]


# --- Hybrid Search ---


def hybrid_search(df, filters, query=None, top_k=5):
    """Combine filter-based and semantic search."""
    filtered_df = filter_movies(df, filters)

    if filtered_df.empty:
        # Try relaxing year constraints
        relaxed_filters = {
            "rating": filters.get("rating"),
            "year": None,
            "year_after": None,
            "year_before": None,
            "actor": filters.get("actor"),
            "director": filters.get("director"),
            "genre": filters.get("genre"),
        }
        relaxed_df = filter_movies(df, relaxed_filters)

        if not relaxed_df.empty:
            return relaxed_df.head(min(top_k, len(relaxed_df)))

        # Fall back to semantic search
        if query:
            return semantic_search(query, top_k)

        return pd.DataFrame()

    if query:
        return semantic_search_filtered(filtered_df, query, top_k)

    return filtered_df.head(min(top_k, len(filtered_df)))


# --- Query Classification ---


def is_movie_query(query):
    """Check if the query is related to movies."""
    query_embedding = model.encode([query])
    score = cosine_similarity(query_embedding, domain_embedding)[0][0]
    return score > 0.20


def extract_top_k(query):
    """Extract how many movies the user wants."""
    match = re.search(r"(\d+)\s+(?:movies|films|recommendations|results)", query.lower())
    if match:
        k = int(match.group(1))
        return min(k, 20)  # cap at 20
    if "few" in query.lower():
        return 3
    return 5


# --- Conversation Memory ---


def get_default_memory():
    return {
        "rating": None,
        "year": None,
        "year_after": None,
        "year_before": None,
        "actor": None,
        "director": None,
        "genre": None,
    }


def update_memory(old_filters, new_filters):
    """Merge new filters into existing memory."""
    updated = old_filters.copy()

    if new_filters.get("actor") is not None:
        updated["actor"] = new_filters["actor"]
        updated["director"] = None

    elif new_filters.get("director") is not None:
        updated["director"] = new_filters["director"]
        updated["actor"] = None

    for key in ["rating", "year", "year_after", "year_before", "genre"]:
        if new_filters.get(key) is not None:
            updated[key] = new_filters[key]

    return updated


def is_new_query(query):
    """Detect if user is starting a new search vs. a follow-up."""
    q = query.lower()
    new_keywords = [
        "recommend", "suggest", "find", "show me", "search",
        "movies", "movie", "films", "what are", "give me",
        "list", "top",
    ]
    return any(word in q for word in new_keywords)


def is_followup_query(query, memory):
    """Detect if this is a follow-up to a previous query."""
    if not any(memory.values()):
        return False

    followup_patterns = [
        r"^(what about|how about|and|also|more|another)",
        r"^(any|some)\s+(more|other)",
        r"^(with|by|from|in)\s+",
    ]
    q = query.lower().strip()
    for pattern in followup_patterns:
        if re.match(pattern, q):
            return True
    return False


# --- Response Formatting ---


def format_movie_result(movie, idx):
    """Format a single movie result with details."""
    title = movie.get("title", "Unknown")
    year = movie.get("year", "")
    rating = movie.get("vote_average", 0)
    genres = movie.get("genres", [])

    year_str = f" ({int(year)})" if pd.notna(year) else ""
    rating_str = f"⭐ {rating:.1f}" if pd.notna(rating) and rating > 0 else ""
    genre_str = ", ".join(genres[:3]) if isinstance(genres, list) else ""

    line = f"{idx}. **{title}**{year_str}"
    details = []
    if rating_str:
        details.append(rating_str)
    if genre_str:
        details.append(genre_str)
    if details:
        line += f"  \n   {' | '.join(details)}"

    return line


def format_results(results, filters):
    """Format search results into a response string."""
    if results.empty:
        return "No movies found matching your criteria. Try broadening your search."

    # Build intro
    parts = []
    if filters.get("actor"):
        parts.append(f"featuring **{filters['actor'].title()}**")
    if filters.get("director"):
        parts.append(f"directed by **{filters['director'].title()}**")
    if filters.get("genre"):
        parts.append(f"in the **{filters['genre'].title()}** genre")
    if filters.get("year"):
        parts.append(f"from **{int(filters['year'])}**")
    if filters.get("year_after"):
        parts.append(f"after **{int(filters['year_after'])}**")
    if filters.get("year_before"):
        parts.append(f"before **{int(filters['year_before'])}**")
    if filters.get("rating"):
        parts.append(f"with rating above **{filters['rating']}**")

    if parts:
        intro = f"Here are movies {', '.join(parts)}:"
    else:
        intro = "Here are some movies you might enjoy:"

    lines = [intro, ""]
    for i, (_, movie) in enumerate(results.iterrows(), 1):
        lines.append(format_movie_result(movie, i))

    return "\n".join(lines)


# --- Main Chatbot Function ---


def chatbot(query, memory=None, last_results=None):
    """
    Process a user query and return a response.

    Args:
        query: User's input text
        memory: Conversation memory (filter state from previous queries)
        last_results: DataFrame of the last set of results

    Returns:
        tuple: (response_text, updated_memory, current_results)
    """
    if memory is None:
        memory = get_default_memory()

    # Handle "explain" / "tell me more" queries
    if any(word in query.lower() for word in ["explain", "tell me more", "describe", "about"]):
        if last_results is not None and not last_results.empty:
            # Determine which movie
            ordinals = {
                "first": 0, "1st": 0,
                "second": 1, "2nd": 1,
                "third": 2, "3rd": 2,
                "fourth": 3, "4th": 3,
                "fifth": 4, "5th": 4,
                "last": -1,
            }
            idx = 0
            for word, pos in ordinals.items():
                if word in query.lower():
                    idx = pos
                    break

            # Also match by number: "explain 3"
            num_match = re.search(r"(\d+)", query)
            if num_match:
                idx = int(num_match.group(1)) - 1

            if idx == -1:
                idx = len(last_results) - 1

            idx = max(0, min(idx, len(last_results) - 1))
            movie = last_results.iloc[idx]

            title = movie.get("title", "Unknown")
            overview = movie.get("overview", "No description available.")
            year = movie.get("year", "")
            rating = movie.get("vote_average", 0)
            genres = movie.get("genres", [])

            year_str = f" ({int(year)})" if pd.notna(year) else ""
            rating_str = f"⭐ {rating:.1f}/10" if pd.notna(rating) and rating > 0 else ""
            genre_str = ", ".join(genres[:5]) if isinstance(genres, list) else ""

            response = f"### {title}{year_str}\n\n"
            if rating_str:
                response += f"{rating_str}"
            if genre_str:
                response += f" | {genre_str}"
            response += f"\n\n{overview}"

            return response, memory, last_results

        return "I don't have a previous movie list to explain. Ask me for some recommendations first!", memory, last_results

    # Handle greetings
    greetings = ["hi", "hello", "hey", "howdy", "greetings"]
    if query.lower().strip() in greetings:
        return (
            "Hey! I'm your movie recommendation bot. Ask me things like:\n"
            "- *\"comedy movies after 2015\"*\n"
            "- *\"movies directed by Christopher Nolan\"*\n"
            "- *\"sci-fi movies with rating above 8\"*\n"
            "- *\"movies starring Leonardo DiCaprio\"*",
            memory,
            last_results,
        )

    # Check if movie-related
    if not is_movie_query(query) and not is_followup_query(query, memory):
        return (
            "I specialize in movie recommendations! Try asking something like:\n"
            "- *\"action movies after 2010\"*\n"
            "- *\"movies with Tom Hanks\"*\n"
            "- *\"thriller movies with rating above 7\"*",
            memory,
            last_results,
        )

    # Reset memory for new queries
    if is_new_query(query) and not is_followup_query(query, memory):
        memory = get_default_memory()

    # Extract filters (LLM first, regex fallback)
    filters = extract_filters_llm(query)
    if not any(filters.values()):
        filters = extract_filters(query)

    # Normalize names
    if filters.get("actor"):
        filters["actor"] = filters["actor"].strip().lower()
    if filters.get("director"):
        filters["director"] = filters["director"].strip().lower()
    if filters.get("genre"):
        filters["genre"] = filters["genre"].strip().lower()

    # Update memory
    memory = update_memory(memory, filters)

    top_k = extract_top_k(query)

    # Determine search strategy
    has_structured_filter = any([
        memory.get("rating"),
        memory.get("year"),
        memory.get("year_after"),
        memory.get("year_before"),
        memory.get("actor"),
        memory.get("director"),
        memory.get("genre"),
    ])

    if has_structured_filter:
        results = hybrid_search(movies, memory, None, top_k)
    else:
        results = hybrid_search(movies, memory, query, top_k)

    if isinstance(results, pd.DataFrame) and results.empty:
        return "No movies found matching your criteria. Try a different search!", memory, None

    response = format_results(results, memory)
    return response, memory, results
