"""
fetch_article.py — Fetch full article content from a URL.

Uses requests + BeautifulSoup to scrape the main text body from a news link.
Called by nodes (e.g. ingestion or llm_classifier) when a URL is provided
instead of raw article text.
"""

import requests
from bs4 import BeautifulSoup


def fetch_article_content(url: str) -> str:
    """
    Fetch and extract the main article text from a given URL.

    Args:
        url: The full URL of the news article.

    Returns:
        The extracted article body text as a single string.
        Returns an empty string if the fetch fails.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise: scripts, styles, navs, footers
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text from <p> tags (most common for article bodies)
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return text

    except Exception as e:
        print(f"[fetch_article] Failed to fetch {url}: {e}")
        return ""


def is_url(text: str) -> bool:
    """Check if the input looks like a URL rather than article text."""
    return text.strip().startswith(("http://", "https://"))
