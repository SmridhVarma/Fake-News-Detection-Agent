"""
verification_tools.py — LangChain @tool wrappers and logic for verification-stage skills.
"""

import os
import requests
from langchain_core.tools import tool

def search_related_articles(query: str, page_size: int = 5) -> list[dict]:
    """Search for related articles using NewsAPI."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key: return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "pageSize": page_size, "sortBy": "relevancy", "language": "en", "apiKey": api_key}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for article in data.get("articles", []):
            articles.append({"title": article.get("title", ""), "source": article.get("source", {}).get("name", "Unknown"), "description": article.get("description", ""), "url": article.get("url", "")})
        return articles
    except Exception as e:
        print(f"[cross_reference] Error searching NewsAPI: {e}")
        return []

def cross_reference_article(article_title: str, article_text: str) -> dict:
    """Cross-reference an article's claims by searching for related coverage."""
    query = article_title if article_title else " ".join(article_text.split()[:15])
    related = search_related_articles(query, page_size=5)
    sources = list(set(a["source"] for a in related if a["source"]))
    count = len(related)
    if count == 0: summary = "No related articles found. This could indicate the story is exclusive, too niche, or fabricated."
    elif count <= 2: summary = f"Only {count} related article(s) found from {', '.join(sources)}. Limited corroboration available."
    else: summary = f"Found {count} related articles from {len(sources)} source(s): {', '.join(sources)}. Multiple sources corroborate this story."
    return {"related_count": count, "sources": sources, "corroboration_summary": summary, "related_articles": related}

@tool
def cross_reference_tool(article_title: str, article_text: str) -> str:
    """Search for related news articles to cross-reference and verify claims.
    Uses NewsAPI to find corroborating or contradicting coverage.
    Returns a summary of how many related sources were found."""
    result = cross_reference_article(article_title, article_text)
    lines = [f"Related articles found: {result['related_count']}", f"Sources: {', '.join(result['sources']) if result['sources'] else 'None'}", f"Summary: {result['corroboration_summary']}"]
    for i, art in enumerate(result["related_articles"][:3], 1): lines.append(f"  [{i}] {art['source']}: {art['title']}")
    return "\n".join(lines)
