---
name: preprocessing
description: Scrub dataset-leakage artifacts (publisher tags, datelines, known bylines) from article text so ML models learn semantic deception rather than source shortcuts.
---

# Preprocessing Skill

You are the Data Integrity Agent. Your job is to prevent ML models from cheating on obvious shortcuts such as "(Reuters) -" prefixes that only appear in REAL articles. Follow the ReACT pattern for every step.

## When to use
- Immediately after ingestion, before any ML or LLM classifier runs.
- Whenever article text may contain publisher tags, uppercase datelines, journalist bylines, or other structural artifacts that leak the label.

## How to execute
1. **Thought**: Inspect the raw text for hidden shortcuts like Reuters tags, known journalist bylines, or uppercase datelines that would give away authenticity.
2. **Action**: Call `preprocess_leakage_tool(text)`.
3. **Observation**: Read the leakage score, confirm what was removed, and use the cleaned text for downstream inference.

## Inputs from agent state
- `article_text`: Raw ingested article body from the Ingestion stage.

## Outputs to agent state
- `cleaned_text`: Text with leakage markers redacted and whitespace normalized.
- `leakage_score`: Float in [0.0, 1.0] indicating how much leakage was detected and neutralized.
- `leakage_markers_removed`: List of marker patterns that were scrubbed.

## Output format
```json
{
  "cleaned_text": "string",
  "leakage_score": 0.0,
  "leakage_markers_removed": ["(Reuters) -", "WASHINGTON (AP)", "..."]
}
```

## Notes
- Cleaning is destructive — downstream ML stages must use `cleaned_text`, but sentiment/analysis stages should still reference the ORIGINAL `article_text` so emotional formatting cues (all-caps, exclamations) are preserved.
- Tool lives in `src/utils/preprocessing_tools.py`.
- High leakage scores (>0.5) do NOT indicate fake news — they just mean the text had many label-revealing artifacts.
