"""
Shared helpers for tag normalization, slugging, and noise filtering.
"""
from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

GENERIC_DROP_TOKENS = {
    "all", "and", "annual", "a", "as", "by", "chart", "com", "data", "dataset",
    "daily", "direct", "district", "document", "email", "file", "gmail", "gov",
    "govt", "india", "json", "latlong", "line", "map", "monthly", "name",
    "national", "nadu", "nic", "number", "pdf", "proper", "rank", "sample",
    "state", "stacked", "table", "test", "total", "weekly", "wise", "year",
    "yearly", "xml",
}

KEEP_SHORT_TOKENS = {
    "ab", "aids", "bpl", "cpi", "csv", "css", "db", "gdp", "gst", "hiv", "iit",
    "inc", "it", "iti", "ivr", "law", "me", "ncr", "nic", "ngo", "nra", "obc",
    "pdf", "phc", "pmo", "rbi", "rti", "sc", "st",
}

GEOGRAPHIC_PHRASES = {
    "andaman and nicobar islands",
    "andhra pradesh",
    "arunachal pradesh",
    "dadra and nagar haveli",
    "daman and diu",
    "delhi",
    "goa",
    "gujarat",
    "himachal pradesh",
    "jammu and kashmir",
    "karnataka",
    "kerala",
    "lakshadweep",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "puducherry",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
}


def normalize_tag_text(value: str) -> str:
    """Normalize a raw tag candidate to a lowercase canonical phrase."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tag_slug(value: str) -> str:
    """Convert a raw or normalized tag into a stable slug."""
    normalized = normalize_tag_text(value)
    if not normalized:
        return ""
    return _NON_ALNUM_RE.sub("-", normalized).strip("-")


def is_noise_tag(value: str) -> bool:
    """Return True if the candidate is clearly not useful as a canonical tag."""
    normalized = normalize_tag_text(value)
    if not normalized:
        return True

    if normalized in GENERIC_DROP_TOKENS:
        return True

    if normalized in GEOGRAPHIC_PHRASES:
        return False

    if len(normalized) < 3 and normalized not in KEEP_SHORT_TOKENS:
        return True

    if normalized.isdigit():
        return True

    if any(part.isdigit() for part in normalized.split()):
        return True

    if normalized in {"com", "org", "net", "www", "http", "https"}:
        return True

    return False

