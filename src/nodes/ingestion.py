"""
ingestion_node — Receives raw article text and preprocesses it.
"""

from src.state import AgentState
from src.utils.preprocessing import clean_text_for_traditional_ml, clean_text_for_transformers
from skills.calculate_features import calculate_article_scores

def ingestion_node(state: AgentState) -> dict:
    """Clean and prepare the article text for downstream nodes."""
    
    # 1. Get raw input (usually passed in when you invoke the graph)
    raw_text = state.get("raw_input", "")
    
    # 2. Clean text using two different pipelines
    cleaned_llm = clean_text_for_transformers(raw_text)
    cleaned_ml = clean_text_for_traditional_ml(raw_text)
    
    # 3. Call the Skill from the skills directory
    # Run features on the LLM text so casing/punctuation is preserved for stylistic checks!
    features = calculate_article_scores(raw_text) #Changed from cleaned_ml to raw_text so that stylistic features are more accurate (based on original text, as cleaning might remove important stylistic cues)
    
    # 4. Return the updates to the AgentState
    return {
        "article_text": raw_text,               # legacy mapping defaults to LLM version
        "article_text_llm": cleaned_llm,           # Explicit transformer-ready text
        "article_text_ml": cleaned_ml,             # Explicit traditional ML text
        "word_count": len(cleaned_ml.split()),     # ML word count is more statistically pure
        "caps_ratio": features["caps_ratio"],
        "style_score": features["sub_variance"],
        "mean_subjectivity": features["mean_subjectivity"],
        "lexical_density": features["lexical_density"],
        "has_dateline": features["has_dateline"]
    }
