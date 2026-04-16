---
name: ingestion
description: Safely extract and structure incoming article data (URL or raw text) and compute baseline stylistic features before downstream processing.
mode: organisational
---

# Ingestion Skill

You are the Ingestion Agent. Your job is to safely extract and structure incoming data before it flows downstream into the pipeline. Follow the ReACT pattern (Thought → Action → Observation) for every step.

## When to use
- At the very start of the pipeline, whenever a new article enters the system.
- When the input is either a raw URL or pasted article text and must be normalized into a single `article_text` field.
- Before any analysis, preprocessing, or classification stage runs.

## How to execute
1. **Thought**: Determine whether the input is a URL or raw text.
2. **Action**:
   - If a URL is provided, call `fetch_url_tool(url)` to scrape the article body.
   - Once article text is available, immediately call `calculate_features_tool(text)` on the unadulterated text to capture stylistic formatting signals (caps ratio, lexical density, subjectivity variance, dateline presence).
3. **Observation**: Record the scraped text and numerical feature dictionary.

## Inputs from agent state
- `raw_url` (optional): URL string if user submitted a link.
- `raw_text` (optional): Pasted article text if user submitted plain text.
- At least one of the two must be present.

## Outputs to agent state
- `article_text`: Original article body (retained for LLM and stylistic analysis).
- `article_text_ml`: Lowercased, punctuation-stripped version for TF-IDF inference.
- `word_count`: Token count of the ML-cleaned text.
- `caps_ratio`: Fraction of ALL-CAPS words (≥3 chars) in the raw text.
- `style_score`: Subjectivity variance across sentences (proxy for writing style consistency).
- `mean_subjectivity`: Mean TextBlob subjectivity score across sentences.
- `lexical_density`: Ratio of unique tokens to total tokens.
- `has_dateline`: Boolean — whether the article begins with an uppercase dateline (e.g. "WASHINGTON —").
- `source_domain` (optional): Domain extracted from the URL if URL input was provided.

## Output format
```json
{
  "article_text": "string",
  "article_text_ml": "string",
  "word_count": 0,
  "caps_ratio": 0.0,
  "style_score": 0.0,
  "mean_subjectivity": 0.0,
  "lexical_density": 0.0,
  "has_dateline": false,
  "source_domain": "string or null"
}
```

## Notes
- Always run `calculate_features_tool` on the **unmodified** text, before any leakage scrubbing in the Preprocessing stage — otherwise stylistic markers get lost.
- If `fetch_url_tool` fails, fall back to any user-pasted text; never silently drop the request.
- Tools live in `src/utils/ingestion_tools.py`.
