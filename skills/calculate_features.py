import re
import statistics
from textblob import TextBlob

def calculate_article_scores(text: str) -> dict:
    """
	Takes a string and returns raw numerical features.
    """
    if not text or not isinstance(text, str):
        return {
            "sub_variance": 0, "mean_subjectivity": 0, 
            "lexical_density": 0, "caps_ratio": 0, "has_dateline": False
        }

    blob = TextBlob(text)
    sentences = blob.sentences
    words = text.lower().split()
    
    # 1. Subjectivity Metrics
    #Measures tone consistency - Real news has low variance
    sent_subjectivity = [s.sentiment.subjectivity for s in sentences]
    sub_variance = statistics.variance(sent_subjectivity) if len(sent_subjectivity) > 1 else 0
    mean_subjectivity = blob.sentiment.subjectivity
    
    # 2. Structural Dateline Detection
    # Looks for the pattern: "CITY (Agency)"
    dateline_pattern = r'^[A-Z]{2,}\s?\(.*?\)\s?\-\s?' 
    has_standard_dateline = bool(re.search(dateline_pattern, text[:100]))
    
    # 3. Lexical Density 
    # (Measures vocabulary richness - Fake news often uses repetitive trigger words)
    lexical_density = len(set(words)) / len(words) if len(words) > 0 else 0
    
    # 4. Caps Ratio
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    return {
        "sub_variance": round(sub_variance, 4),
        "mean_subjectivity": round(mean_subjectivity, 4),
        "lexical_density": round(lexical_density, 4),
        "caps_ratio": round(caps_ratio, 4),
        "has_dateline": has_standard_dateline
    }