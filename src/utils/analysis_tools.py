"""
analysis_tools.py — LangChain @tool wrappers and logic for analysis-stage skills.
"""

from langchain_core.tools import tool
from textblob import TextBlob
from urllib.parse import urlparse

# ─── Sentiment Analysis ────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """Performs detailed sentiment analysis on article text."""
    if not text or not isinstance(text, str):
        return {"overall_polarity": 0.0, "overall_subjectivity": 0.0, "sentence_count": 0, 
                "exclamation_ratio": 0.0, "caps_word_ratio": 0.0, "emotional_tone": "neutral", "tone_score": 0.0}

    blob = TextBlob(text)
    sentences = blob.sentences
    words = text.split()

    overall_polarity = blob.sentiment.polarity
    overall_subjectivity = blob.sentiment.subjectivity

    exclaim_sentences = sum(1 for s in sentences if str(s).strip().endswith("!"))
    exclamation_ratio = exclaim_sentences / len(sentences) if sentences else 0.0

    caps_words = sum(1 for w in words if w.isupper() and len(w) >= 3)
    caps_word_ratio = caps_words / len(words) if words else 0.0

    tone_score = min((0.4 * overall_subjectivity) + (0.3 * min(exclamation_ratio * 2, 1.0)) + (0.3 * min(caps_word_ratio * 5, 1.0)), 1.0)
    tone_score = round(tone_score, 4)

    emotional_tone = "neutral" if tone_score < 0.25 else "slightly_biased" if tone_score < 0.50 else "highly_emotional"

    return {
        "overall_polarity": round(overall_polarity, 4), "overall_subjectivity": round(overall_subjectivity, 4),
        "sentence_count": len(sentences), "exclamation_ratio": round(exclamation_ratio, 4),
        "caps_word_ratio": round(caps_word_ratio, 4), "emotional_tone": emotional_tone, "tone_score": tone_score
    }

@tool
def sentiment_analysis_tool(text: str) -> str:
    """Analyze the sentiment and emotional tone of an article.
    Detects sensationalism signals like excessive caps, exclamation marks,
    and high subjectivity. Returns a tone assessment."""
    result = analyze_sentiment(text)
    return (
        f"Emotional Tone: {result['emotional_tone']}\n"
        f"Tone Score: {result['tone_score']} (0=neutral, 1=sensationalist)\n"
        f"Polarity: {result['overall_polarity']}\n"
        f"Subjectivity: {result['overall_subjectivity']}\n"
        f"Exclamation Ratio: {result['exclamation_ratio']}\n"
        f"ALL-CAPS Word Ratio: {result['caps_word_ratio']}\n"
        f"Sentence Count: {result['sentence_count']}"
    )

# ─── Source Credibility ────────────────────────────────────────────

CREDIBILITY_DB = {
    "reuters.com": 0.95, "apnews.com": 0.95, "bbc.com": 0.93, "nytimes.com": 0.90, "washingtonpost.com": 0.90,
    "theguardian.com": 0.88, "wsj.com": 0.90, "cnn.com": 0.80, "foxnews.com": 0.65, "nbcnews.com": 0.80,
    "infowars.com": 0.10, "breitbart.com": 0.25, "theonion.com": 0.05, "babylonbee.com": 0.05
}

def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except Exception: return ""

def check_source_credibility(url: str = "", domain: str = "") -> dict:
    if url and not domain: domain = extract_domain(url)
    if not domain: return {"domain": "", "credibility_score": 0.5, "credibility_tier": "unknown", "is_known_source": False, "warning": "No URL provided."}
    
    score = CREDIBILITY_DB.get(domain)
    is_known = score is not None
    score = score if is_known else 0.5
    
    tier = "high" if score >= 0.8 else "moderate" if score >= 0.6 else "low" if score >= 0.2 else "very_low"
    warning = None
    if tier == "moderate": warning = f"Source '{domain}' has moderate credibility. Cross-check claims."
    elif tier == "low": warning = f"Source '{domain}' has low credibility. Treat claims with skepticism."
    elif tier == "very_low": warning = f"Source '{domain}' is a known unreliable site. Do not trust."
    
    return {"domain": domain, "credibility_score": score, "credibility_tier": tier, "is_known_source": is_known, "warning": warning}

@tool
def source_credibility_tool(url: str) -> str:
    """Check the credibility of a news source based on its domain.
    Looks up the domain against a curated database of known reliable
    and unreliable news outlets. Returns a credibility assessment."""
    result = check_source_credibility(url=url)
    lines = [f"Domain: {result['domain']}", f"Credibility Score: {result['credibility_score']}", f"Credibility Tier: {result['credibility_tier']}", f"Known Source: {result['is_known_source']}"]
    if result.get("warning"): lines.append(f"Warning: {result['warning']}")
    return "\n".join(lines)
