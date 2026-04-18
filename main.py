"""
main.py - Gradio UI entry point for Fake News Detection Agent

Launches a Gradio interface that:
  - Accepts article text or a news URL as input
  - Runs the full LangGraph agent pipeline (src/graph.py)
  - Displays: Final Classification, ML Score, LLM Score, Reasoning Summary
"""

import os
import time
import traceback

from dotenv import load_dotenv
import gradio as gr

from src.graph import run_agent

load_dotenv()

EXAMPLE_TEXTS = [
    [
        "WASHINGTON (Reuters) - The U.S. Federal Reserve kept interest rates unchanged on Wednesday "
        "and signaled it still planned three rate cuts for 2024, despite recent data showing inflation "
        "remains sticky. Fed Chair Jerome Powell said during a press conference that policymakers still "
        "believe inflation is on a path back to the central bank's 2% target, but acknowledged the "
        "journey may take longer than initially expected. The decision was widely anticipated by markets, "
        "which rallied following the announcement.",
    ],
    [
        "EXPOSED: Government Officials Caught RED-HANDED Hiding Secret Alien Technology!!! "
        "Whistle-blowers from deep inside the Pentagon have FINALLY come forward to reveal that our "
        "government has been LYING to us for DECADES about contact with extraterrestrial beings!!! "
        "The mainstream media is REFUSING to cover this story because they are CONTROLLED by the very "
        "same elites who want to keep this information from YOU!!! Share this before they DELETE it!!!",
    ],
]

# Custom CSS: override Gradio defaults for verdict display, hide built-in progress overlays,
# and polish input/tab/button styling.
CUSTOM_CSS = """
/* Global typography and spacing */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 16px 0 10px;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 12px;
}
.app-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: var(--body-text-color);
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 15px;
    color: var(--body-text-color-subdued);
    margin: 0;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Section headers */
.section-label {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--body-text-color-subdued);
    margin-bottom: 12px;
}

/* Input area styling */
.input-card {
    background: var(--background-fill-primary);
    border-radius: 12px;
    padding: 14px;
    border: 1px solid var(--border-color-primary);
}
.input-card textarea,
.input-card input[type="text"] {
    background: #ffffff !important;
    color: #000000 !important;
    border-color: var(--border-color-primary) !important;
}
.input-card textarea::placeholder,
.input-card input[type="text"]::placeholder {
    color: var(--body-text-color-subdued) !important;
}

/* Verdict display */
.verdict-container {
    text-align: center;
    padding: 20px 16px;
    border-radius: 14px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
}
.verdict-container.real {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #10b981;
}
.verdict-container.fake {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 2px solid #ef4444;
}
.verdict-container.unknown {
    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    border: 2px solid #9ca3af;
}
.verdict-emoji {
    font-size: 42px;
    display: block;
    margin-bottom: 4px;
}
.verdict-label {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 0 0 4px;
}
.verdict-container.real .verdict-label { color: #059669; }
.verdict-container.fake .verdict-label { color: #dc2626; }
.verdict-container.unknown .verdict-label { color: #6b7280; }
.verdict-confidence {
    font-size: 16px;
    color: var(--body-text-color-subdued);
    margin: 0;
}

/* Score cards */
.score-card {
    background: var(--background-fill-secondary);
    border-radius: 10px;
    padding: 12px;
    border: 1px solid var(--border-color-primary);
    margin-bottom: 8px;
}
.score-card-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--body-text-color-subdued);
    margin-bottom: 6px;
}
.score-card-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--body-text-color);
}

/* Agreement badge */
.agreement-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.agreement-badge.agree {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}
.agreement-badge.disagree {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
}

/* Submit button */
.submit-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
}

.btn-row {
    gap: 10px !important;
}
.btn-row button {
    flex: 1 !important;
}

/* Accordion styling */
.accordion-section {
    border-radius: 10px !important;
    overflow: hidden;
}

/* Hide Gradio queue/progress overlays; use custom status panel instead */
.progress-text,
.loading-status,
[data-testid="status-bar"],
[data-testid="block-progress"] {
    display: none !important;
}

.status-panel {
    background: rgba(37, 99, 235, 0.12);
    border: 1px solid rgba(96, 165, 250, 0.45);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 10px;
}
.status-panel.error-panel {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(248, 113, 113, 0.50);
}
.status-panel p,
.status-panel li,
.status-panel h3 {
    margin: 0;
    line-height: 1.4;
}
.status-panel ul {
    margin-top: 6px;
}
/* Tab styling: selected tab should be prominent, unselected subdued */
.tab-nav button {
    color: var(--body-text-color-subdued) !important;
    font-weight: 500 !important;
}
.tab-nav button.selected {
    color: var(--body-text-color) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--primary-600) !important;
}

/* Input focus polish */
textarea:focus, input:focus {
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 12px 0;
    margin-top: 12px;
    border-top: 1px solid var(--border-color-primary);
    font-size: 12px;
    color: var(--body-text-color-subdued);
}

/* Layout density */
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
}

.compact-row {
    gap: 12px !important;
}

.summary-box textarea {
    min-height: 52px !important;
}

/* Hide only the label text, keep the input visible */
.hide-label .label-wrap {
    display: none !important;
}

/* Hide empty result card before first run */
#result-score-cards:empty {
    display: none !important;
}

/* Disagreement CTA under the badge */
.disagreement-cta {
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
    font-size: 13px;
    font-weight: 500;
    text-align: center;
}

/* URL tab hint */
.url-hint {
    margin-top: 6px;
    font-size: 12px;
    color: var(--body-text-color-subdued);
    line-height: 1.4;
}

/* Labeled example buttons */
.example-header {
    margin-top: 8px;
    font-size: 13px;
    color: var(--body-text-color-subdued);
}
.example-row {
    gap: 8px !important;
    margin-top: 4px;
}
.example-btn {
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    white-space: normal !important;
    text-align: left !important;
}
.example-btn.example-real {
    border-left: 3px solid #10b981 !important;
}
.example-btn.example-fake {
    border-left: 3px solid #ef4444 !important;
}
"""


def _build_verdict_html(final_label: str, final_score: float) -> str:
    """Return an HTML block displaying the verdict label, emoji, and confidence."""
    emoji_map = {"REAL": "\u2705", "FAKE": "\u274c"}
    emoji = emoji_map.get(final_label, "\u2753")
    css_class = final_label.lower() if final_label in ("REAL", "FAKE") else "unknown"
    return (
        f'<div class="verdict-container {css_class}">'
        f'<span class="verdict-emoji">{emoji}</span>'
        f'<h2 class="verdict-label">{final_label}</h2>'
        f'<p class="verdict-confidence">Confidence: {final_score:.0%}</p>'
        f"</div>"
    )


def _build_score_cards(
    ml_label, ml_score, llm_label, llm_score, eval_score, eval_agreement
):
    """Return HTML score cards for ML verdict, LLM verdict, DeepEval score, and agreement badge.

    Args:
        ml_label, ml_score: Traditional ML model output.
        llm_label, llm_score: LLM agent output.
        eval_score: DeepEval reasoning quality (0–1).
        eval_agreement: Whether ML and LLM labels match.
    """
    agree_class = "agree" if eval_agreement else "disagree"
    agree_text = "Agreed" if eval_agreement else "Disagreed"
    agree_icon = "\u2713" if eval_agreement else "\u26a0"
    cta_html = (
        ""
        if eval_agreement
        else '<div class="disagreement-cta">Models disagree — recommend human review before publishing.</div>'
    )
    return (
        f'<div class="score-card">'
        f'<div class="score-card-label">ML Model Verdict</div>'
        f'<div class="score-card-value">{ml_label} <span style="font-size:16px;font-weight:400;color:var(--body-text-color-subdued)">({ml_score:.0%})</span></div>'
        f"</div>"
        f'<div class="score-card">'
        f'<div class="score-card-label">AI Agent Verdict</div>'
        f'<div class="score-card-value">{llm_label} <span style="font-size:16px;font-weight:400;color:var(--body-text-color-subdued)">({llm_score:.0%})</span></div>'
        f"</div>"
        f'<div class="score-card">'
        f'<div class="score-card-label">Reasoning Quality (DeepEval)</div>'
        f'<div class="score-card-value">{eval_score:.2f} <span style="font-size:16px;font-weight:400;color:var(--body-text-color-subdued)">/ 1.00</span></div>'
        f"</div>"
        f'<div style="text-align:center;margin-top:8px;">'
        f'<span class="agreement-badge {agree_class}">{agree_icon} Models {agree_text}</span>'
        f"</div>"
        f"{cta_html}"
    )


def classify_article(text: str, url: str, active_input_type: str):
    """Gradio handler — validates input, runs the pipeline, and yields streaming UI updates.

    This is a generator function. Each yield emits a 6-tuple matching the Gradio outputs:
    (verdict_html, score_cards_html, summary, explanation, details_md, status_md)
    """
    input_text = text.strip() if text else ""
    input_url = url.strip() if url else ""

    if active_input_type == "url":
        if not input_url:
            yield (
                "",
                "",
                "",
                "",
                "",
                gr.update(
                    value="### Ready\n- Enter a news URL, then click Analyze Article",
                    elem_classes=["status-panel", "error-panel"],
                ),
            )
            raise gr.Error("URL tab is selected. Please enter a news URL to analyze.")
        input_type = "url"
        user_input = input_url
    else:
        if not input_text:
            yield (
                "",
                "",
                "",
                "",
                "",
                gr.update(
                    value="### Ready\n- Paste article text, then click Analyze Article",
                    elem_classes=["status-panel", "error-panel"],
                ),
            )
            raise gr.Error(
                "Paste Article tab is selected. Please enter article text to analyze."
            )
        input_type = "text"
        user_input = input_text

    has_v2_artifacts = os.path.exists("./models/v2/training_artifacts.joblib")
    warmup_msg = (
        "Using cached v2 artifacts for faster inference."
        if has_v2_artifacts
        else "No cached artifacts detected. Initial run may train models."
    )

    yield (
        "",
        "",
        "",
        "",
        "",
        gr.update(
            value=f"### Processing\n- Step 1/4: Input validated ({input_type.upper()})\n- Step 2/4: Pipeline started\n- {warmup_msg}",
            elem_classes=["status-panel"],
        ),
    )

    try:
        t0 = time.time()
        result = run_agent(user_input, input_type=input_type)
        elapsed = time.time() - t0
        print(f"\n>>> Pipeline completed in {elapsed:.1f}s")
    except Exception as e:
        print(f">>> Pipeline error: {e}")
        traceback.print_exc()

        err_msg = str(e)
        if input_type == "url":
            yield (
                "",
                "",
                "",
                "",
                "",
                gr.update(
                    value=f"### Error\n- Could not read content from the URL\n- The site may block automated access\n- **Try pasting the article text directly** in the 'Paste Article' tab\n- Detail: {err_msg}",
                    elem_classes=["status-panel", "error-panel"],
                ),
            )
        else:
            yield (
                "",
                "",
                "",
                "",
                "",
                gr.update(
                    value=f"### Error\n- Analysis failed\n- Detail: {err_msg}",
                    elem_classes=["status-panel", "error-panel"],
                ),
            )
        return

    final_label = result.get("final_label", "UNKNOWN")
    final_score = result.get("final_score", 0.0)
    summary = result.get("summary", "No summary available.")
    explanation = result.get("explanation", "")

    # Append corroborating sources from NewsAPI (QW-6) to the explanation markdown
    related_articles = result.get("related_articles", []) or []
    if related_articles:
        top_sources = related_articles[:3]
        sources_md = "\n\n---\n\n### Corroborating Sources (NewsAPI)\n\n"
        for i, art in enumerate(top_sources, 1):
            title = art.get("title") or "(untitled)"
            source = art.get("source") or "Unknown"
            url_link = art.get("url") or ""
            if url_link:
                sources_md += f"{i}. **{source}** — [{title}]({url_link})\n"
            else:
                sources_md += f"{i}. **{source}** — {title}\n"
        total = len(related_articles)
        if total > 3:
            sources_md += f"\n_Showing top 3 of {total} matches._\n"
        explanation = (explanation or "") + sources_md
    else:
        explanation = (explanation or "") + (
            "\n\n---\n\n### Corroborating Sources (NewsAPI)\n\n"
            "_No related articles found. This may indicate an exclusive story, a niche topic, "
            "or fabricated content._\n"
        )

    ml_label = result.get("ml_label", "N/A")
    ml_score = result.get("ml_score", 0.0)
    llm_label = result.get("llm_label", "N/A")
    llm_score = result.get("llm_score", 0.0)
    eval_score = result.get("eval_score", 0.0)
    eval_agreement = result.get("eval_agreement", False)

    # Convert ML raw probability (P(REAL)) to confidence in the predicted label
    ml_confidence = ml_score if ml_label == "REAL" else 1 - ml_score

    verdict_html = _build_verdict_html(final_label, final_score)
    score_cards_html = _build_score_cards(
        ml_label, ml_confidence, llm_label, llm_score, eval_score, eval_agreement
    )

    caps_ratio = result.get("caps_ratio")
    mean_subjectivity = result.get("mean_subjectivity")
    sub_variance = result.get("style_score")
    lexical_density = result.get("lexical_density")
    has_dateline = result.get("has_dateline")

    def _fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v:.2%}" if pct else f"{v:.2f}"

    dateline_str = "n/a" if has_dateline is None else ("Yes" if has_dateline else "No")

    # Build tool-trace section showing which tools the ReACT agent invoked.
    tool_trace = result.get("llm_tool_trace", []) or []
    tool_friendly = {
        "cross_reference_tool": "cross_reference_tool (NewsAPI)",
        "sentiment_analysis_tool": "sentiment_analysis_tool (TextBlob)",
        "source_credibility_tool": "source_credibility_tool (domain tier)",
        "preprocess_leakage_tool": "preprocess_leakage_tool (clean text)",
        "fetch_url_tool": "fetch_url_tool (URL scrape)",
    }
    # Count invocations per tool name
    tool_counts = {}
    for entry in tool_trace:
        tool_counts[entry["name"]] = tool_counts.get(entry["name"], 0) + 1
    all_tool_names = [
        "sentiment_analysis_tool",
        "source_credibility_tool",
        "cross_reference_tool",
        "preprocess_leakage_tool",
        "fetch_url_tool",
    ]
    for name in tool_counts:
        if name not in all_tool_names:
            all_tool_names.append(name)

    if tool_trace:
        trace_lines = [
            "### Agent Tool Trace",
            "",
            "| Tool | Called | Times |",
            "| --- | :---: | :---: |",
        ]
        for name in all_tool_names:
            count = tool_counts.get(name, 0)
            mark = "\u2713" if count > 0 else "\u2014"
            display_name = tool_friendly.get(name, name)
            trace_lines.append(f"| {display_name} | {mark} | {count if count else '-'} |")
        trace_md = "\n".join(trace_lines) + "\n\n"
        # Show cross_reference_tool detail if it ran
        for entry in tool_trace:
            if entry["name"] == "cross_reference_tool":
                query_arg = (entry.get("args") or {}).get("article_title", "") or "(none)"
                first_line = (entry.get("result") or "").splitlines()[0] if entry.get("result") else ""
                trace_md += (
                    f"**cross_reference_tool query:** `{query_arg[:120]}`  \n"
                    f"**Result (first line):** {first_line}\n\n"
                )
                break
    else:
        trace_md = (
            "### Agent Tool Trace\n\n"
            "_No tool calls recorded — the LLM responded without invoking any tools._\n\n"
        )

    details_md = (
        f"### Key Signals\n\n"
        f"- **Caps ratio:** {_fmt(caps_ratio, pct=True)} — sensationalism indicator\n"
        f"- **Subjectivity (mean):** {_fmt(mean_subjectivity)} — opinion vs fact\n"
        f"- **Subjectivity (variance):** {_fmt(sub_variance)} — tonal switching\n"
        f"- **Lexical density:** {_fmt(lexical_density)} — content vs filler\n"
        f"- **Dateline present:** {dateline_str} — publisher-style proxy\n\n"
        f"{trace_md}"
        f"### Pipeline Breakdown\n\n"
        f"**ML Model:** {ml_label} with {ml_confidence:.0%} confidence\n"
        f"**AI Agent:** {llm_label} with {llm_score:.0%} confidence\n"
        f"**Agreement:** {'Yes' if eval_agreement else 'No'}\n"
        f"**DeepEval Score:** {eval_score:.2f} / 1.00\n"
    )

    status_md = (
        "### Processing Complete\n"
        f"- Step 1/4: Input + ingestion\n"
        f"- Step 2/4: ML inference finished ({ml_label}, {ml_confidence:.0%})\n"
        f"- Step 3/4: LLM fact-check finished ({llm_label}, {llm_score:.0%})\n"
        f"- Step 4/4: Evaluation + aggregation complete\n"
        f"- Total runtime: {elapsed:.1f}s"
    )

    yield (
        verdict_html,
        score_cards_html,
        summary,
        explanation,
        details_md,
        gr.update(value=status_md, elem_classes=["status-panel"]),
    )


def _clear_results():
    return (
        "",
        "",
        "",
        "",
        "",
        gr.update(
            value="### Ready\n- Enter article text or URL, then click Analyze Article",
            elem_classes=["status-panel"],
        ),
    )


# Soft theme with slate/blue palette. Light-mode-first with dark-mode overrides.
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    spacing_size="md",
    radius_size="lg",
    text_size=gr.themes.sizes.text_lg,
).set(
    body_background_fill="#ffffff",
    body_background_fill_dark="#0f172a",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#1e293b",
    background_fill_secondary="#f8fafc",
    background_fill_secondary_dark="#334155",
    border_color_primary="#e2e8f0",
    border_color_primary_dark="#475569",
    block_title_text_weight="600",
    block_label_text_weight="500",
    block_label_text_size="*text_sm",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_large_padding="16px 32px",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#1e293b",
    input_border_color="#e2e8f0",
    input_border_color_dark="#475569",
    input_border_width="1px",
    input_radius="*radius_md",
    input_padding="12px 16px",
    block_radius="*radius_lg",
    block_border_width="1px",
    block_padding="20px",
)

import joblib

def _load_dashboard_data():
    try:
        if not os.path.exists("./models/v2/training_artifacts.joblib"):
            return [], "UNKNOWN"
        artifacts = joblib.load("./models/v2/training_artifacts.joblib")
        results = artifacts.get("candidate_validation_results", {})
        table_data = []
        for model_name, metrics in results.items():
            table_data.append([
                model_name.replace("_", " ").title(),
                f"{metrics.get('accuracy', 0):.1%}",
                f"{metrics.get('precision', 0):.1%}",
                f"{metrics.get('recall', 0):.1%}",
                f"{metrics.get('f1', 0):.1%}"
            ])
        return table_data, artifacts.get("selected_model_name", "UNKNOWN").replace("_", " ").title()
    except Exception as e:
        print(f"Failed to load dashboard data: {e}")
        return [], "UNKNOWN"

DASHBOARD_TABLE_DATA, BEST_MODEL_NAME = _load_dashboard_data()


with gr.Blocks(
    title="Fake News Detection Agent",
) as demo:
    with gr.Column(elem_classes="app-header"):
        gr.HTML(
            "<h1>Fake News Detection Agent</h1>"
            "<p>An AI-powered tool that combines machine learning with autonomous fact-checking "
            "to determine whether a news article is <strong>REAL</strong> or <strong>FAKE</strong>.</p>"
        )

    with gr.Row(elem_classes="compact-row"):
        with gr.Column(scale=6, min_width=420):
            gr.HTML('<div class="section-label">Input</div>')
            active_input_type = gr.State("text")
            with gr.Column(elem_classes="input-card"):
                with gr.Tabs() as input_tabs:
                    with gr.Tab("Paste Article", id="text") as text_tab:
                        text_input = gr.Textbox(
                            label="Article Text",
                            placeholder="Paste the full article text here...",
                            lines=8,
                            show_label=False,
                            elem_classes="hide-label",
                        )
                    with gr.Tab("News URL", id="url") as url_tab:
                        url_input = gr.Textbox(
                            label="Article URL",
                            placeholder="https://www.example.com/news/article",
                            show_label=False,
                            elem_classes="hide-label",
                        )
                        gr.Markdown(
                            "_Note: major publishers (Reuters, NYT, WSJ, Bloomberg) block "
                            "automated scraping. If the URL fails, paste the article text "
                            "into the **Paste Article** tab instead._",
                            elem_classes="url-hint",
                        )

            with gr.Row(elem_classes="btn-row"):
                submit_btn = gr.Button(
                    "Analyze Article",
                    variant="primary",
                    size="lg",
                    elem_classes="submit-btn",
                )

                clear_btn = gr.Button(
                    "Clear Results",
                    variant="secondary",
                    size="lg",
                    elem_classes="submit-btn",
                )

            text_tab.select(lambda: "text", None, active_input_type)
            url_tab.select(lambda: "url", None, active_input_type)

            gr.Markdown("**Try an example:**", elem_classes="example-header")
            with gr.Row(elem_classes="example-row"):
                real_example_btn = gr.Button(
                    "Real: Reuters Fed announcement",
                    size="sm",
                    elem_classes="example-btn example-real",
                )
                fake_example_btn = gr.Button(
                    "Fake: Sensationalist conspiracy",
                    size="sm",
                    elem_classes="example-btn example-fake",
                )
            real_example_btn.click(lambda: EXAMPLE_TEXTS[0][0], None, text_input)
            fake_example_btn.click(lambda: EXAMPLE_TEXTS[1][0], None, text_input)

        with gr.Column(scale=6, min_width=420):
            gr.HTML('<div class="section-label">Result</div>')

            status_output = gr.Markdown(
                value="### Ready\n- Enter article text or URL, then click Analyze Article",
                elem_classes="status-panel",
            )

            verdict_output = gr.HTML()

            with gr.Row():
                with gr.Column():
                    score_cards_output = gr.HTML(elem_id="result-score-cards")

            summary_output = gr.Textbox(
                label="Summary",
                interactive=False,
                lines=2,
                show_label=False,
                elem_classes=["hide-label", "summary-box"],
            )

            with gr.Accordion(
                "Detailed Analysis", open=False, elem_classes="accordion-section"
            ):
                details_output = gr.Markdown()

            with gr.Accordion(
                "Full AI Explanation", open=False, elem_classes="accordion-section"
            ):
                explanation_output = gr.Markdown()

            with gr.Accordion(
                "Model Performance Dashboard", open=False, elem_classes="accordion-section"
            ):
                gr.Markdown(
                    "_Trained on Kaggle Fake-and-Real News Dataset (2015–2018). "
                    "Retraining recommended before production deployment._"
                )
                gr.Markdown(
                    f"**Training Insight:** The ML Verdict was generated by the **{BEST_MODEL_NAME}** model, "
                    f"which proved to be the most accurate during training. See full validation metrics below."
                )
                gr.Dataframe(
                    headers=["Model", "Accuracy", "Precision", "Recall", "F1 Score"],
                    value=DASHBOARD_TABLE_DATA,
                    interactive=False
                )
                with gr.Row():
                    gr.Image(value="evaluation_outputs/v2_detailed/roc_curve_comparison.png", label="ROC Curves (Test Dataset)", height=400)
                    gr.Image(value="evaluation_outputs/v2_detailed/confusion_matrix_neural_network.png", label="Neural Network Confusion Matrix (Selected Model)", height=400)

    submit_btn.click(
        fn=classify_article,
        inputs=[text_input, url_input, active_input_type],
        outputs=[
            verdict_output,
            score_cards_output,
            summary_output,
            explanation_output,
            details_output,
            status_output,
        ],
        show_progress="hidden",
    )

    clear_btn.click(
        fn=_clear_results,
        inputs=None,
        outputs=[
            verdict_output,
            score_cards_output,
            summary_output,
            explanation_output,
            details_output,
            status_output,
        ],
    )

    gr.HTML(
        '<div class="app-footer">'
        "Built with <strong>LangGraph</strong> + <strong>Scikit-learn</strong> + "
        "<strong>OpenAI GPT-4o-mini</strong> + <strong>DeepEval</strong>"
        "</div>"
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
                allowed_paths=["evaluation_outputs"],
    )
