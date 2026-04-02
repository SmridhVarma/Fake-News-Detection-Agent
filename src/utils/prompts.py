"""
prompts.py — Prompt templates for LLM-based classification.
"""

FACT_CHECK_PROMPT = """You are a fact-checking expert. Analyze the following article and determine
whether it is REAL or FAKE news.

Article:
{article_text}

Respond in the following JSON format:
{{
    "label": "REAL" or "FAKE",
    "confidence": 0.0 to 1.0,
    "reasoning": "Your detailed explanation here"
}}
"""
