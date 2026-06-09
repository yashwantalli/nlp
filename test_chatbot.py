"""
Comprehensive unit tests for chatbot.py

chatbot.py performs heavy work at module level (loads CSV, initializes
SentenceTransformer, computes embeddings).  We patch those before the import
so that tests stay fast and don't require a GPU or model download.
"""

import sys
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Test data – 5 movies with columns matching cleaned_movies.csv
# cast/crew/genres/keywords are in the *raw* string form that convert() and
# get_director() will parse during module-level initialization.
# ---------------------------------------------------------------------------
_NUM_MOVIES = 5
_EMBED_DIM = 384

_TEST_DF = pd.DataFrame(
    {
        "budget": [237000000, 185000000, 160000000, 35000000, 165000000],
        "genres": [
            "['Action', 'Adventure', 'Fantasy']",
            "['Action', 'Crime', 'Drama']",
            "['Action', 'Science Fiction', 'Adventure']",
            "['Comedy']",
            "['Adventure', 'Drama', 'Science Fiction']",
        ],
        "keywords": [
            "['space', 'war', 'alien']",
            "['hero', 'villain', 'gotham']",
            "['dream', 'subconscious', 'heist']",
            "['vegas', 'bachelor party', 'hangover']",
            "['space', 'time travel', 'wormhole']",
        ],
        "overview": [
            "A marine is sent to an alien moon.",
            "A vigilante fights crime in Gotham.",
            "A thief enters dreams to steal secrets.",
            "Three friends wake up after a wild bachelor party.",
            "Explorers travel through a wormhole in space.",
        ],
        "popularity": [150.4, 140.2, 130.0, 100.5, 120.3],
        "runtime": [162.0, 152.0, 148.0, 100.0, 169.0],
        "title": [
            "Avatar",
            "The Dark Knight",
            "Inception",
            "The Hangover",
            "Interstellar",
        ],
        "vote_average": [7.2, 9.0, 8.8, 7.6, 8.6],
        "cast": [
            "['Sam Worthington', 'Zoe Saldana', 'Sigourney Weaver']",
            "['Christian Bale', 'Heath Ledger', 'Aaron Eckhart']",
            "['Leonardo DiCaprio', 'Joseph Gordon-Levitt']",
            "['Bradley Cooper', 'Ed Helms', 'Zach Galifianakis']",
            "['Matthew McConaughey', 'Anne Hathaway']",
        ],
        "crew": [
            "[{'name': 'James Cameron', 'job': 'Director'}]",
            "[{'name': 'Christopher Nolan', 'job': 'Director'}]",
            "[{'name': 'Christopher Nolan', 'job': 'Director'}]",
            "[{'name': 'Todd Phillips', 'job': 'Director'}]",
            "[{'name': 'Christopher Nolan', 'job': 'Director'}]",
        ],
        "year": [2009.0, 2008.0, 2010.0, 2009.0, 2014.0],
        "combined": [
            "A marine is sent to an alien moon. Action Adventure Fantasy space war alien Sam Worthington Zoe Saldana",
            "A vigilante fights crime in Gotham. Action Crime Drama hero villain gotham Christian Bale Heath Ledger",
            "A thief enters dreams to steal secrets. Action Science Fiction dream subconscious heist Leonardo DiCaprio",
            "Three friends wake up after a wild bachelor party. Comedy vegas bachelor party hangover Bradley Cooper",
            "Explorers travel through a wormhole in space. Adventure Drama Science Fiction space time travel Matthew McConaughey",
        ],
    }
)

# ---------------------------------------------------------------------------
# Deterministic mock embeddings – so cosine similarity is reproducible
# ---------------------------------------------------------------------------
np.random.seed(42)
_MOVIE_EMBEDDINGS = np.random.rand(_NUM_MOVIES, _EMBED_DIM).astype(np.float32)
_DOMAIN_EMBEDDING = np.random.rand(1, _EMBED_DIM).astype(np.float32)

_call_count = {"n": 0}


def _mock_encode(texts, **kwargs):
    """Return deterministic embeddings based on input length."""
    if isinstance(texts, str):
        texts = [texts]
    n = len(texts)
    if n == _NUM_MOVIES:
        return _MOVIE_EMBEDDINGS
    if n == 1:
        # Return a deterministic vector seeded from the text content
        seed = sum(ord(c) for c in str(texts[0])) % 10000
        rng = np.random.RandomState(seed)
        return rng.rand(1, _EMBED_DIM).astype(np.float32)
    return np.random.rand(n, _EMBED_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# Patch heavy module-level operations BEFORE importing chatbot
# ---------------------------------------------------------------------------
_mock_model = MagicMock()
_mock_model.encode = MagicMock(side_effect=_mock_encode)

_pd_read_csv_patch = patch("pandas.read_csv", return_value=_TEST_DF.copy())
_st_patch = patch(
    "sentence_transformers.SentenceTransformer", return_value=_mock_model
)

_pd_read_csv_patch.start()
_st_patch.start()

# Remove chatbot from sys.modules if previously imported (safety)
sys.modules.pop("chatbot", None)

import chatbot  # noqa: E402 – must come after patches
from chatbot import (  # noqa: E402
    convert,
    get_director,
    extract_filters,
    extract_top_k,
    filter_movies,
    update_memory,
    reset_memory,
    is_new_query,
    query_llm,
    extract_filters_llm,
    hybrid_search,
    semantic_search,
    semantic_search_filtered,
    is_movie_query,
    is_followup_query,
)


# ===================================================================
# TESTS: convert()
# ===================================================================
class TestConvert:
    def test_list_of_dicts(self):
        text = "[{'name': 'Action'}, {'name': 'Comedy'}]"
        assert convert(text) == ["Action", "Comedy"]

    def test_list_of_strings(self):
        text = "['Action', 'Comedy']"
        assert convert(text) == ["Action", "Comedy"]

    def test_empty_list(self):
        assert convert("[]") == []

    def test_invalid_string(self):
        assert convert("not a list") == []

    def test_none_input(self):
        assert convert(None) == []

    def test_numeric_value(self):
        assert convert(123) == []

    def test_nested_dicts_with_name_key(self):
        text = "[{'name': 'Sci-Fi', 'id': 1}, {'name': 'Thriller', 'id': 2}]"
        assert convert(text) == ["Sci-Fi", "Thriller"]

    def test_single_element_list(self):
        assert convert("['Drama']") == ["Drama"]


# ===================================================================
# TESTS: get_director()
# ===================================================================
class TestGetDirector:
    def test_single_director(self):
        text = "[{'name': 'James Cameron', 'job': 'Director'}]"
        assert get_director(text) == ["James Cameron"]

    def test_multiple_crew_members(self):
        text = (
            "[{'name': 'James Cameron', 'job': 'Director'}, "
            "{'name': 'Jon Landau', 'job': 'Producer'}]"
        )
        assert get_director(text) == ["James Cameron"]

    def test_no_director(self):
        text = "[{'name': 'Jon Landau', 'job': 'Producer'}]"
        assert get_director(text) == []

    def test_multiple_directors(self):
        text = (
            "[{'name': 'Joel Coen', 'job': 'Director'}, "
            "{'name': 'Ethan Coen', 'job': 'Director'}]"
        )
        assert get_director(text) == ["Joel Coen", "Ethan Coen"]

    def test_empty_list(self):
        assert get_director("[]") == []

    def test_invalid_string(self):
        assert get_director("not valid") == []

    def test_already_list(self):
        data = [{"name": "Nolan", "job": "Director"}]
        assert get_director(data) == ["Nolan"]

    def test_none_input(self):
        assert get_director(None) == []

    def test_list_of_strings(self):
        # Strings in the list are not dicts, should return []
        text = "['Christopher Nolan']"
        assert get_director(text) == []


# ===================================================================
# TESTS: extract_filters()
# ===================================================================
class TestExtractFilters:
    def _defaults(self):
        return {
            "rating": None,
            "year": None,
            "year_after": None,
            "year_before": None,
            "actor": None,
            "director": None,
        }

    def test_rating_above(self):
        f = extract_filters("movies with rating above 8")
        assert f["rating"] == 8.0

    def test_rating_greater_than(self):
        f = extract_filters("rating greater than 7.5")
        assert f["rating"] == 7.5

    def test_exact_year(self):
        f = extract_filters("movies from 2010")
        assert f["year"] == 2010

    def test_year_after(self):
        f = extract_filters("movies after 2015")
        assert f["year_after"] == 2015

    def test_year_before(self):
        f = extract_filters("movies before 2000")
        assert f["year_before"] == 2000

    def test_actor_with_keyword(self):
        f = extract_filters("movies with leonardo dicaprio")
        assert f["actor"] is not None
        assert "leonardo dicaprio" in f["actor"].lower()

    def test_actor_starring(self):
        f = extract_filters("movies starring tom hanks")
        assert f["actor"] is not None
        assert "tom hanks" in f["actor"].lower()

    def test_director(self):
        f = extract_filters("movies directed by christopher nolan")
        assert f["director"] is not None
        assert "christopher nolan" in f["director"].lower()

    def test_no_filters(self):
        f = extract_filters("recommend me something good")
        assert all(v is None for v in f.values())

    def test_multiple_filters(self):
        f = extract_filters("movies with rating above 8 from 2010")
        assert f["rating"] == 8.0
        assert f["year"] == 2010

    def test_year_1900s(self):
        f = extract_filters("classic movies from 1994")
        assert f["year"] == 1994


# ===================================================================
# TESTS: extract_top_k()
# ===================================================================
class TestExtractTopK:
    def test_with_number(self):
        assert extract_top_k("show me 10 movies") == 10

    def test_with_films(self):
        assert extract_top_k("top 3 films") == 3

    def test_default(self):
        assert extract_top_k("show me some movies please") == 5

    def test_no_number(self):
        assert extract_top_k("good action stuff") == 5

    def test_large_number(self):
        assert extract_top_k("give me 100 movies") == 100


# ===================================================================
# TESTS: filter_movies()
# ===================================================================
class TestFilterMovies:
    @pytest.fixture
    def sample_df(self):
        """Post-transform DataFrame matching what filter_movies expects."""
        return pd.DataFrame(
            {
                "title": [
                    "Avatar",
                    "The Dark Knight",
                    "Inception",
                    "The Hangover",
                    "Interstellar",
                ],
                "vote_average": [7.2, 9.0, 8.8, 7.6, 8.6],
                "year": [2009.0, 2008.0, 2010.0, 2009.0, 2014.0],
                "popularity": [150.4, 140.2, 130.0, 100.5, 120.3],
                "cast": [
                    ["sam worthington", "zoe saldana", "sigourney weaver"],
                    ["christian bale", "heath ledger", "aaron eckhart"],
                    ["leonardo dicaprio", "joseph gordon-levitt"],
                    ["bradley cooper", "ed helms", "zach galifianakis"],
                    ["matthew mcconaughey", "anne hathaway"],
                ],
                "crew": [
                    ["james cameron"],
                    ["christopher nolan"],
                    ["christopher nolan"],
                    ["todd phillips"],
                    ["christopher nolan"],
                ],
            }
        )

    def _null_filters(self, **overrides):
        f = {
            "rating": None,
            "year": None,
            "year_after": None,
            "year_before": None,
            "actor": None,
            "director": None,
        }
        f.update(overrides)
        return f

    def test_filter_by_rating(self, sample_df):
        result = filter_movies(sample_df, self._null_filters(rating=8.0))
        assert all(result["vote_average"] >= 8.0)
        assert "Avatar" not in result["title"].values

    def test_filter_by_exact_year(self, sample_df):
        result = filter_movies(sample_df, self._null_filters(year=2009.0))
        assert set(result["title"].values) == {"Avatar", "The Hangover"}

    def test_filter_by_year_after(self, sample_df):
        result = filter_movies(sample_df, self._null_filters(year_after=2010.0))
        assert "Inception" in result["title"].values
        assert "Interstellar" in result["title"].values
        assert "The Dark Knight" not in result["title"].values

    def test_filter_by_year_before(self, sample_df):
        result = filter_movies(sample_df, self._null_filters(year_before=2010.0))
        assert "Avatar" in result["title"].values
        assert "The Dark Knight" in result["title"].values
        assert "Interstellar" not in result["title"].values

    def test_filter_by_actor_exact(self, sample_df):
        result = filter_movies(
            sample_df, self._null_filters(actor="leonardo dicaprio")
        )
        assert "Inception" in result["title"].values

    def test_filter_by_actor_fuzzy(self, sample_df):
        result = filter_movies(
            sample_df, self._null_filters(actor="heath ledger")
        )
        assert "The Dark Knight" in result["title"].values

    def test_filter_by_director(self, sample_df):
        result = filter_movies(
            sample_df, self._null_filters(director="christopher nolan")
        )
        titles = result["title"].tolist()
        assert "The Dark Knight" in titles
        assert "Inception" in titles
        assert "Interstellar" in titles

    def test_filter_no_match(self, sample_df):
        result = filter_movies(
            sample_df, self._null_filters(actor="nonexistent actor")
        )
        assert len(result) == 0

    def test_no_filters(self, sample_df):
        result = filter_movies(sample_df, self._null_filters())
        assert len(result) == len(sample_df)

    def test_combined_rating_and_year(self, sample_df):
        result = filter_movies(
            sample_df, self._null_filters(rating=8.5, year=2010.0)
        )
        assert set(result["title"].values) == {"Inception"}

    def test_director_results_sorted(self, sample_df):
        result = filter_movies(
            sample_df, self._null_filters(director="christopher nolan")
        )
        # When director filter is applied, results are sorted by vote_average desc
        ratings = result["vote_average"].tolist()
        assert ratings == sorted(ratings, reverse=True)


# ===================================================================
# TESTS: update_memory()
# ===================================================================
class TestUpdateMemory:
    def _empty_memory(self):
        return {
            "rating": None,
            "year": None,
            "year_after": None,
            "year_before": None,
            "actor": None,
            "director": None,
        }

    def test_update_actor_clears_director(self):
        old = self._empty_memory()
        old["director"] = "nolan"
        new = {"actor": "dicaprio", "director": None, "rating": None,
               "year": None, "year_after": None, "year_before": None}
        result = update_memory(old, new)
        assert result["actor"] == "dicaprio"
        assert result["director"] is None

    def test_update_director_clears_actor(self):
        old = self._empty_memory()
        old["actor"] = "dicaprio"
        new = {"actor": None, "director": "nolan", "rating": None,
               "year": None, "year_after": None, "year_before": None}
        result = update_memory(old, new)
        assert result["director"] == "nolan"
        assert result["actor"] is None

    def test_update_rating(self):
        old = self._empty_memory()
        new = {"actor": None, "director": None, "rating": 8.0,
               "year": None, "year_after": None, "year_before": None}
        result = update_memory(old, new)
        assert result["rating"] == 8.0

    def test_preserves_existing_values(self):
        old = self._empty_memory()
        old["rating"] = 7.0
        new = {"actor": None, "director": None, "rating": None,
               "year": 2020, "year_after": None, "year_before": None}
        result = update_memory(old, new)
        assert result["rating"] == 7.0
        assert result["year"] == 2020

    def test_update_year_after(self):
        old = self._empty_memory()
        new = {"actor": None, "director": None, "rating": None,
               "year": None, "year_after": 2015, "year_before": None}
        result = update_memory(old, new)
        assert result["year_after"] == 2015


# ===================================================================
# TESTS: reset_memory()
# ===================================================================
class TestResetMemory:
    def test_resets_all_fields(self):
        chatbot.memory["actor"] = "dicaprio"
        chatbot.memory["rating"] = 8.0
        reset_memory()
        assert all(v is None for v in chatbot.memory.values())

    def test_reset_idempotent(self):
        reset_memory()
        reset_memory()
        assert all(v is None for v in chatbot.memory.values())


# ===================================================================
# TESTS: is_new_query()
# ===================================================================
class TestIsNewQuery:
    @pytest.mark.parametrize(
        "query",
        [
            "show me some movies",
            "action movie",
            "films with great actors",
            "who is the actor in inception",
            "best director",
            "movies with high rating",
            "comedy genre",
        ],
    )
    def test_movie_keywords_detected(self, query):
        assert is_new_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "hello there",
            "what is the weather",
            "tell me a joke",
        ],
    )
    def test_non_movie_queries(self, query):
        assert is_new_query(query) is False


# ===================================================================
# TESTS: query_llm() – mocked HTTP
# ===================================================================
class TestQueryLLM:
    @patch("chatbot.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value.json.return_value = [
            {"generated_text": "This is a movie about adventure."}
        ]
        result = query_llm("describe this movie")
        assert result == "This is a movie about adventure."

    @patch("chatbot.requests.post")
    def test_non_list_response(self, mock_post):
        mock_post.return_value.json.return_value = {"error": "model loading"}
        result = query_llm("describe")
        assert result == ""

    @patch("chatbot.requests.post")
    def test_exception(self, mock_post):
        mock_post.side_effect = Exception("network error")
        result = query_llm("test")
        assert result == ""


# ===================================================================
# TESTS: extract_filters_llm() – mocked LLM
# ===================================================================
class TestExtractFiltersLLM:
    @patch("chatbot.query_llm")
    def test_valid_json(self, mock_llm):
        mock_llm.return_value = json.dumps(
            {
                "rating": 8.0,
                "year": None,
                "year_after": 2010,
                "year_before": None,
                "actor": "tom hanks",
                "director": None,
            }
        )
        result = extract_filters_llm("tom hanks movies after 2010 rating 8")
        assert result["rating"] == 8.0
        assert result["actor"] == "tom hanks"
        assert result["year_after"] == 2010

    @patch("chatbot.query_llm")
    def test_invalid_json_fallback(self, mock_llm):
        mock_llm.return_value = "not json at all"
        result = extract_filters_llm("movies directed by christopher nolan")
        # Should fall back to regex extraction
        assert result["director"] is not None
        assert "christopher nolan" in result["director"].lower()

    @patch("chatbot.query_llm")
    def test_empty_response_fallback(self, mock_llm):
        mock_llm.return_value = ""
        result = extract_filters_llm("movies with rating above 7")
        # Falls back to regex
        assert result["rating"] == 7.0


# ===================================================================
# TESTS: semantic_search()
# ===================================================================
class TestSemanticSearch:
    def test_returns_correct_number(self):
        results = semantic_search("action adventure", top_k=3)
        assert len(results) == 3

    def test_returns_dataframe(self):
        results = semantic_search("sci-fi space movie")
        assert isinstance(results, pd.DataFrame)
        assert "title" in results.columns

    def test_top_k_larger_than_dataset(self):
        results = semantic_search("test", top_k=100)
        assert len(results) == _NUM_MOVIES

    def test_top_k_one(self):
        results = semantic_search("space", top_k=1)
        assert len(results) == 1


# ===================================================================
# TESTS: semantic_search_filtered()
# ===================================================================
class TestSemanticSearchFiltered:
    def test_returns_subset(self):
        subset = chatbot.movies.head(3).copy()
        results = semantic_search_filtered(subset, "action movie", top_k=2)
        assert len(results) == 2

    def test_returns_dataframe(self):
        subset = chatbot.movies.copy()
        results = semantic_search_filtered(subset, "comedy", top_k=2)
        assert isinstance(results, pd.DataFrame)

    def test_top_k_exceeds_filtered(self):
        subset = chatbot.movies.head(2).copy()
        results = semantic_search_filtered(subset, "test", top_k=10)
        assert len(results) == 2


# ===================================================================
# TESTS: hybrid_search()
# ===================================================================
class TestHybridSearch:
    def _null_filters(self, **overrides):
        f = {
            "rating": None,
            "year": None,
            "year_after": None,
            "year_before": None,
            "actor": None,
            "director": None,
        }
        f.update(overrides)
        return f

    def test_with_rating_filter(self):
        results = hybrid_search(
            chatbot.movies, self._null_filters(rating=8.5), None, 5
        )
        assert isinstance(results, pd.DataFrame)
        assert all(results["vote_average"] >= 8.5)

    def test_with_query_no_filters(self):
        results = hybrid_search(
            chatbot.movies, self._null_filters(), "space adventure", 3
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

    def test_no_match_fallback_to_semantic(self):
        results = hybrid_search(
            chatbot.movies,
            self._null_filters(actor="nonexistent person xyz"),
            "space",
            3,
        )
        # Should fall back to semantic or relaxed search and return results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_returns_limited_results(self):
        results = hybrid_search(
            chatbot.movies, self._null_filters(), None, 2
        )
        assert len(results) <= 2


# ===================================================================
# TESTS: is_movie_query()
# ===================================================================
class TestIsMovieQuery:
    def test_movie_related(self):
        result = is_movie_query("recommend me a good action movie")
        assert isinstance(result, (bool, np.bool_))

    def test_returns_bool(self):
        result = is_movie_query("what is the weather today")
        assert isinstance(result, (bool, np.bool_))


# ===================================================================
# TESTS: is_followup_query()
# ===================================================================
class TestIsFollowupQuery:
    def test_no_memory_returns_false(self):
        reset_memory()
        assert is_followup_query("tell me more") is False

    def test_with_memory(self):
        chatbot.memory["actor"] = "nolan"
        result = is_followup_query("what about their other movies")
        assert isinstance(result, (bool, np.bool_))
        # Clean up
        reset_memory()

    def test_empty_memory_values(self):
        reset_memory()
        result = is_followup_query("more like that")
        assert result is False


# ===================================================================
# TESTS: chatbot() – main entry point
# ===================================================================
class TestChatbot:
    def setup_method(self):
        """Reset global state before each test."""
        reset_memory()
        chatbot.last_results = None

    @patch("chatbot.extract_filters_llm")
    def test_explain_first_movie(self, mock_llm_filters):
        mock_llm_filters.return_value = {
            "rating": None, "year": None, "year_after": None,
            "year_before": None, "actor": None, "director": None,
        }
        # First, set last_results to a known DataFrame
        chatbot.last_results = pd.DataFrame(
            {
                "title": ["Avatar", "Inception"],
                "overview": ["An alien moon adventure.", "A dream heist."],
            }
        )
        result = chatbot.chatbot("explain the first movie")
        assert "Avatar" in result

    @patch("chatbot.extract_filters_llm")
    def test_explain_second_movie(self, mock_llm_filters):
        mock_llm_filters.return_value = {
            "rating": None, "year": None, "year_after": None,
            "year_before": None, "actor": None, "director": None,
        }
        chatbot.last_results = pd.DataFrame(
            {
                "title": ["Avatar", "Inception"],
                "overview": ["An alien moon adventure.", "A dream heist."],
            }
        )
        result = chatbot.chatbot("explain the second movie")
        assert "Inception" in result

    @patch("chatbot.extract_filters_llm")
    def test_explain_no_previous_results(self, mock_llm_filters):
        mock_llm_filters.return_value = {
            "rating": None, "year": None, "year_after": None,
            "year_before": None, "actor": None, "director": None,
        }
        chatbot.last_results = None
        result = chatbot.chatbot("explain the first movie")
        # Source uses Unicode right single quote (U+2019)
        assert "previous movie" in result.lower() or isinstance(result, list)

    @patch("chatbot.is_movie_query", return_value=True)
    @patch("chatbot.extract_filters_llm")
    def test_movie_query_returns_string_or_list(self, mock_llm, mock_is_movie):
        mock_llm.return_value = {
            "rating": 8.0, "year": None, "year_after": None,
            "year_before": None, "actor": None, "director": None,
        }
        result = chatbot.chatbot("movies with rating above 8")
        assert isinstance(result, (str, list))

    @patch("chatbot.is_movie_query", return_value=False)
    @patch("chatbot.is_followup_query", return_value=False)
    def test_non_movie_query(self, mock_followup, mock_is_movie):
        result = chatbot.chatbot("what is the capital of France")
        # Should return a message about only handling movie queries
        assert isinstance(result, (str, list))
        if isinstance(result, list):
            assert any("movie" in str(r).lower() for r in result)


# ===================================================================
# TESTS: edge cases and integration-ish scenarios
# ===================================================================
class TestEdgeCases:
    def test_convert_with_empty_string(self):
        assert convert("") == []

    def test_get_director_empty_string(self):
        assert get_director("") == []

    def test_extract_filters_empty_string(self):
        f = extract_filters("")
        assert all(v is None for v in f.values())

    def test_extract_top_k_no_digits(self):
        assert extract_top_k("show me movies") == 5

    def test_filter_movies_empty_df(self):
        empty = pd.DataFrame(
            columns=["title", "vote_average", "year", "popularity", "cast", "crew"]
        )
        filters = {
            "rating": None, "year": None, "year_after": None,
            "year_before": None, "actor": None, "director": None,
        }
        result = filter_movies(empty, filters)
        assert len(result) == 0

    def test_update_memory_all_none(self):
        old = {k: None for k in ["rating", "year", "year_after", "year_before", "actor", "director"]}
        new = {k: None for k in ["rating", "year", "year_after", "year_before", "actor", "director"]}
        result = update_memory(old, new)
        assert all(v is None for v in result.values())

    def test_extract_filters_decimal_rating(self):
        f = extract_filters("movies with rating above 7.5")
        assert f["rating"] == 7.5

    def test_extract_filters_year_2000(self):
        f = extract_filters("movies from 2000")
        assert f["year"] == 2000

    def test_convert_dict_without_name_key(self):
        text = "[{'id': 1, 'value': 'test'}]"
        # list of dicts but first dict has no 'name' key → KeyError caught?
        # convert checks isinstance(data[0], dict) then does data[0]['name']
        # This will raise KeyError, caught by bare except → returns []
        result = convert(text)
        assert result == []

    def test_filter_movies_actor_short_name(self):
        """Short actor names should use stricter fuzzy matching (>90 threshold)."""
        df = pd.DataFrame(
            {
                "title": ["Movie A"],
                "vote_average": [7.0],
                "year": [2020.0],
                "popularity": [100.0],
                "cast": [["zoe"]],
                "crew": [["director x"]],
            }
        )
        filters = {
            "rating": None, "year": None, "year_after": None,
            "year_before": None, "actor": "zoe", "director": None,
        }
        result = filter_movies(df, filters)
        assert len(result) == 1

    def test_filter_movies_actor_not_list(self):
        """If cast column is not a list, match_actor should return False."""
        df = pd.DataFrame(
            {
                "title": ["Movie A"],
                "vote_average": [7.0],
                "year": [2020.0],
                "popularity": [100.0],
                "cast": ["not a list"],
                "crew": [["director x"]],
            }
        )
        filters = {
            "rating": None, "year": None, "year_after": None,
            "year_before": None, "actor": "someone", "director": None,
        }
        result = filter_movies(df, filters)
        assert len(result) == 0

    def test_filter_movies_director_not_list(self):
        """If crew column is not a list, match_director should return False."""
        df = pd.DataFrame(
            {
                "title": ["Movie A"],
                "vote_average": [7.0],
                "year": [2020.0],
                "popularity": [100.0],
                "cast": [["actor a"]],
                "crew": ["not a list"],
            }
        )
        filters = {
            "rating": None, "year": None, "year_after": None,
            "year_before": None, "actor": None, "director": "someone",
        }
        result = filter_movies(df, filters)
        assert len(result) == 0
