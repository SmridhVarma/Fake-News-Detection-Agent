---
name: llm_classification
description: Run the ReAct LLM fact-checking agent — analyze tone and source credibility, cross-reference claims via NewsAPI, then compose an evidence-grounded REAL/FAKE verdict with written reasoning for the Gradio UI.
mode: llm_driven
---

# LLM Classification Skill

You are the Fact-Checking Agent. Your job is to verify the authenticity of an article using a ReAct (Thought → Action → Observation) loop, gathering tone, source, and corroboration evidence before emitting a final verdict and written justification.

This stage is decomposed into three internal sub-steps — **Analysis**, **Verification**, and **Explanation** — executed sequentially within a single ReAct agent invocation.

## When to use
- At inference time, immediately after Preprocessing.
- In parallel with (or alongside) the ML Classification stage.
- Once per article — never looped.

## How to execute

### Step 1 — Analysis (tone + source credibility)
1. **Thought**: What non-factual signals can I extract?
2. **Action**:
   - Call `sentiment_analysis_tool(text)` on the **original** article text to capture tone_score, exclamation_ratio, caps_word_ratio, subjectivity.
   - If a URL or domain is available, call `source_credibility_tool(url)` to look up the domain's credibility tier.
3. **Observation**: Record tone metrics and credibility tier.

### Step 2 — Verification (cross-reference)
1. **Thought**: What are the 1-2 most distinctive, verifiable claims?
2. **Action**: Call `cross_reference_tool(article_title_or_query, article_text)` with a concise query derived from the headline or core claim.
3. **Observation**: Record the count of related articles and the list of corroborating source domains.

### Step 3 — Explanation (compose verdict)
1. **Thought**: Group observed signals into (a) stylistic/structural cues, (b) source credibility, (c) independent corroboration.
2. **Action**: Do NOT invoke new tools. Compose the written justification from existing observations only.
3. **Observation**: Emit the final JSON block with label, confidence, and reasoning.

## Inputs from agent state
- `article_text`: Original article body (pre-leakage-scrub, so emotional cues are preserved).
- `raw_url` or `source_domain` (optional): Used for credibility lookup.
- `article_title` (optional): Used to seed the cross-reference query.

## Outputs to agent state
- `llm_label`: `"REAL"` or `"FAKE"`.
- `llm_score`: Confidence in [0.0, 1.0].
- `llm_reasoning`: 3-6 sentence written justification consumed by the Aggregator and displayed in the Gradio UI.

## Output format
The agent must emit a fenced JSON block as its final message:
```json
{
  "label": "REAL",
  "confidence": 0.87,
  "reasoning": "3-6 sentence explanation citing at least one numerical signal (e.g. tone_score, caps_ratio), the source credibility tier (if a URL was provided), and the cross-reference result. Explicitly link each piece of evidence to the verdict and acknowledge any conflicting signals."
}
```

## Notes
- Logic lives in `src/nodes/llm_classifier.py`. Tools live in `src/utils/analysis_tools.py` and `src/utils/verification_tools.py`.
- **Tone signals** must be computed on the original text, not the leakage-scrubbed text, so that ALL-CAPS and exclamation cues survive.
- **Source credibility** requires the raw URL — the caller must forward it into the HumanMessage. Unknown domains default to `credibility_score = 0.5` / tier `"low"`; treat that as "no signal", not negative evidence.
- **Zero NewsAPI results** do NOT automatically imply fabrication — niche, regional, or non-English outlets may be absent from NewsAPI. Flag ambiguity in the reasoning instead of defaulting to FAKE.
- The `reasoning` field must:
  - Be 3-6 sentences — concise but complete.
  - Cite at least one numerical signal.
  - Cite the source credibility tier when a URL was provided.
  - Cite the cross-reference count.
  - Acknowledge conflicting signals and state which evidence dominated.
  - Avoid hedging ("might be", "possibly") unless confidence < 0.6.
- This is the **business-facing textual output** that the Gradio UI surfaces via `aggregator_node.explanation`.
