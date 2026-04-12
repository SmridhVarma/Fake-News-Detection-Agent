"""
preprocessing_tools.py — LangChain @tool wrappers and logic for dataset leakage mitigation.

Addresses the issue where Kaggle fake news datasets contain specific publisher markers
(like 'WASHINGTON (Reuters) -') in real news but not in fake news, artificially inflating ML accuracy.
"""

from langchain_core.tools import tool
import re

def detect_data_leakage(text: str) -> dict:
    """Scans for explicit dataset biases and leakage markers."""
    if not text:
        return {"has_reuters_tag": False, "has_location_prefix": False, "has_twitter_handle": False, "leakage_score": 0.0}
        
    has_reuters_tag = bool(re.search(r'\(Reuters\)', text, re.IGNORECASE))
    
    # Matches patterns like "WASHINGTON -" or "NEW YORK (Reuters) -"
    has_location_prefix = bool(re.search(r'^[A-Z\s]{3,20}\s?(?:\([^)]+\))?\s?[-—]', text[:100]))
    
    has_twitter_handle = bool(re.search(r'@[A-Za-z0-9_]{1,15}', text))
    
    leakage_score = 0.0
    if has_reuters_tag: leakage_score += 0.6
    if has_location_prefix: leakage_score += 0.3
    if has_twitter_handle: leakage_score += 0.1
    
    return {
        "has_reuters_tag": has_reuters_tag,
        "has_location_prefix": has_location_prefix, 
        "has_twitter_handle": has_twitter_handle,
        "leakage_score": min(leakage_score, 1.0)
    }

def strip_publisher_patterns(text: str) -> str:
    """Aggressively redacts publisher names, locations, and watermarks to prevent ML cheating."""
    if not text: return ""
    
    # 1. Remove the location + agency dateline prefix (e.g. "WASHINGTON (Reuters) - ")
    text = re.sub(r'^[A-Z\s]{3,20}\s?(?:\([^)]+\))?\s?[-—]\s?', '', text)
    
    # 2. Remove explicit mentions of Reuters in parentheses
    text = re.sub(r'\(Reuters\)', '', text, flags=re.IGNORECASE)
    
    # 3. Remove "Photo by Getty Images" or similar credits
    text = re.sub(r'Photo by [A-Za-z\s]+(/[A-Za-z\s]+)?', '', text, flags=re.IGNORECASE)
    
    # 4. Remove obvious "Read more:" or "Click here" links that might bias
    text = re.sub(r'Read more(:|\s).*', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def normalize_artifacts(text: str) -> str:
    """Strips excessive symbols that simplify detection of fake news."""
    if not text: return ""
    
    # Replace multiple question marks/exclamation points with a single one
    text = re.sub(r'([!?]){2,}', r'\1', text)
    
    # Replace repeated capitalization blocks with regular text (optional, but good for robust ML)
    # Just removing extra whitespace and symbols for now.
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()

@tool
def preprocess_leakage_tool(text: str) -> str:
    """Detect and remove dataset leakage markers (e.g. publisher datelines, Reuters tags).
    Use this to ensure the text is clean and unbiased before passing it to ML models."""
    leakage_info = detect_data_leakage(text)
    if leakage_info["leakage_score"] > 0:
        cleaned_text = strip_publisher_patterns(text)
        cleaned_text = normalize_artifacts(cleaned_text)
        return (
            f"Leakage Detected: Score {leakage_info['leakage_score']}\n"
            f"Cleaned Text:\n{cleaned_text}"
        )
    return "No significant leakage markers detected. Text is safe for ML."
