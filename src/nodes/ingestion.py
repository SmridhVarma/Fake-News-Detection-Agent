"""
ingestion_node — Receives raw article text and preprocesses it.
"""

from src.state import AgentState
from src.utils.preprocessing import clean_text
from src.utils.ingestion_tools import calculate_article_scores, fetch_article_from_url
from src.utils.analysis_tools import analyze_sentiment, check_source_credibility

def ingestion_node(state: AgentState) -> dict:
    """Clean and prepare the article text for downstream nodes."""
    print("\n>>> [NODE] Starting Ingestion Node...")
    
    # 1. Get raw input (usually passed in when you invoke the graph)
    input_type = state.get("input_type", "text")
    raw_input = state.get("raw_input", "")
    
    if input_type == "url":
        raw_text = fetch_article_from_url(raw_input)
        if not raw_text or len(raw_text.strip()) < 80:
            raise ValueError(
                "Could not extract enough article text from the URL. "
                "Try pasting the article text directly if the site blocks scraping."
            )
    else:
        raw_text = raw_input
    
    # 2. Clean text
    cleaned_ml = clean_text(raw_text)
    
    # 3. Call the Skill from the skills directory
    # Run features on the LLM text so casing/punctuation is preserved for stylistic checks!
    features = calculate_article_scores(raw_text) #Changed from cleaned_ml to raw_text so that stylistic features are more accurate (based on original text, as cleaning might remove important stylistic cues)
    
    # 4. Run sentiment analysis skill
    _ = analyze_sentiment(raw_text)
    
    # 5. Check source credibility if URL was provided
    source_domain = None
    credibility = {}
    if input_type == "url":
        credibility = check_source_credibility(url=raw_input)
        source_domain = credibility.get("domain")
    
    # 5. Return the updates to the AgentState
    result = {
        "article_text": raw_text,
        "article_text_ml": cleaned_ml,             # Explicit traditional ML text
        "word_count": len(cleaned_ml.split()),     # ML word count is more statistically pure
        "caps_ratio": features["caps_ratio"],
        "style_score": features["sub_variance"],
        "mean_subjectivity": features["mean_subjectivity"],
        "lexical_density": features["lexical_density"],
        "has_dateline": features["has_dateline"],
        "source_domain": source_domain,
    }
    print(">>> [NODE] Finished Ingestion Node.")
    return result
