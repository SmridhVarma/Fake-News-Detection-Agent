"""
ingestion_tools.py — LangChain @tool wrappers and logic for ingestion-stage skills.
"""

from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup
import re
import statistics
from textblob import TextBlob

# ─── URL Fetching ──────────────────────────────────────────────────

def fetch_article_from_url(url: str) -> str:
    """Fetches the main textual content from a given URL with multi-strategy extraction."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    }
    try:
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "button"]):
            tag.extract()

        main_content = soup.find('article')
        if not main_content:
            for selector in ['main', '[role="main"]', '#content', '#main-content', '.article-body', '.story-body']:
                main_content = soup.select_one(selector)
                if main_content: break
        if not main_content: main_content = soup.body
        if not main_content: return ""

        paragraphs = main_content.find_all('p')
        text_blocks = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
        if text_blocks: return " ".join(text_blocks)
        return main_content.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"[fetch_url] Error fetching {url}: {e}")
        return ""

@tool
def fetch_url_tool(url: str) -> str:
    """Fetch the main text content from a news article URL.
    Use this when the user provides a URL instead of raw text.
    Returns the extracted article text."""
    text = fetch_article_from_url(url)
    if not text:
        return "ERROR: Could not extract text from the provided URL."
    return text

# ─── Feature Calculation ───────────────────────────────────────────

def calculate_article_scores(text: str) -> dict:
    """Takes a string and returns raw numerical features."""
    if not text or not isinstance(text, str):
        return {"sub_variance": 0, "mean_subjectivity": 0, "lexical_density": 0, "caps_ratio": 0, "has_dateline": False}
    blob = TextBlob(text)
    sentences = blob.sentences
    words = text.lower().split()
    
    sent_subjectivity = [s.sentiment.subjectivity for s in sentences]
    sub_variance = statistics.variance(sent_subjectivity) if len(sent_subjectivity) > 1 else 0
    mean_subjectivity = blob.sentiment.subjectivity
    
    dateline_pattern = r'^[A-Z]{2,}\s?\(.*?\)\s?\-\s?' 
    has_dateline = bool(re.search(dateline_pattern, text[:100]))
    
    lexical_density = len(set(words)) / len(words) if len(words) > 0 else 0
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    return {
        "sub_variance": round(sub_variance, 4),
        "mean_subjectivity": round(mean_subjectivity, 4),
        "lexical_density": round(lexical_density, 4),
        "caps_ratio": round(caps_ratio, 4),
        "has_dateline": has_dateline
    }

@tool
def calculate_features_tool(text: str) -> str:
    """Calculate stylistic and structural features of an article.
    Includes subjectivity variance, lexical density, caps ratio, and dateline detection.
    These help distinguish real vs fake news formatting."""
    result = calculate_article_scores(text)
    return (
        f"Subjectivity Variance: {result['sub_variance']}\n"
        f"Mean Subjectivity: {result['mean_subjectivity']}\n"
        f"Lexical Density: {result['lexical_density']}\n"
        f"Caps Ratio: {result['caps_ratio']}\n"
        f"Has Dateline: {result['has_dateline']}"
    )