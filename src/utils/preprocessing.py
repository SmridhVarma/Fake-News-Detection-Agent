"""
preprocessing.py — Text cleaning helpers.

Example helper pattern:
  - Pure functions, no state dependency
  - Called by nodes, never call nodes
"""

import re


def clean_text(text: str) -> str:
    """Remove extra whitespace, URLs, and normalize casing."""
    text = text.strip()
    text = re.sub(r"http\S+", "", text)           # strip URLs
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    return text
