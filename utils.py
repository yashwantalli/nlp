import ast

from rapidfuzz import fuzz


QUERY_SUFFIX = "movie plot story theme emotion character journey drama"

DEFAULT_FILTERS = {
    "rating": None,
    "year": None,
    "year_after": None,
    "year_before": None,
    "actor": None,
    "director": None,
}


def make_filters(**overrides):
    """Return a fresh default-filters dict, optionally overriding specific keys."""
    filters = DEFAULT_FILTERS.copy()
    filters.update(overrides)
    return filters


def fuzzy_match_name(target, names):
    """Return True if *target* fuzzy-matches any entry in *names*.

    Uses the same threshold logic previously duplicated in match_actor / match_director.
    """
    if not isinstance(names, list):
        return False

    target = target.lower().strip()
    threshold = 90 if len(target) <= 4 else 80

    for name in names:
        name = name.lower().strip()
        if target == name:
            return True
        if fuzz.partial_ratio(target, name) > threshold:
            return True

    return False


def convert(text):
    """Parse a stringified list of dicts (with 'name' keys) or plain list."""
    try:
        data = ast.literal_eval(text)

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return [i["name"] for i in data]

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def get_director(text):
    """Extract director names from a stringified crew list."""
    try:
        data = text if isinstance(text, list) else ast.literal_eval(text)
        return [
            i["name"]
            for i in data
            if isinstance(i, dict) and i.get("job") == "Director"
        ]
    except Exception:
        return []
