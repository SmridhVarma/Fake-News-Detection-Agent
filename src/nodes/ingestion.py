"""
ingestion_node — Receives raw article text and preprocesses it.
"""

from src.state import AgentState
from src.utils.preprocessing import clean_text
# Ensure the filename 'calculate_score' matches exactly what's in your skills/ folder
from skills.calculate_score import calculate_article_scores 

def ingestion_node(state: AgentState) -> dict:
    """Clean and prepare the article text for downstream nodes."""
    
    # 1. Get raw input (usually passed in when you invoke the graph)
    raw_text = state.get("raw_input", "")
    
    # 2. Clean the text (removes HTML, extra whitespace, etc.)
    cleaned = clean_text(raw_text)
    
    # 3. Call the Skill from the skills directory
    # This turns raw text into the numbers your ML model needs
    features = calculate_article_scores(cleaned)
    
    # 4. Return the updates to the AgentState
    # LangGraph will automatically merge these keys into your global state
    return {
        "article_text": cleaned,
        "word_count": len(cleaned.split()),
        "caps_ratio": features["caps_ratio"],
        "style_score": features["sub_variance"],
        "mean_subjectivity": features["mean_subjectivity"],
        "lexical_density": features["lexical_density"],
        "has_dateline": features["has_dateline"]
    }
