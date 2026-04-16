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
.progress-text + .progress-text,
.wrap.svelte-1ipelgc,
.pending.svelte-1ipelgc,
.loading-status,
.status-bar,
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

    ml_label = result.get("ml_label", "N/A")
    ml_score = result.get("ml_score", 0.0)
    llm_label = result.get("llm_label", "N/A")
    llm_score = result.get("llm_score", 0.0)
    eval_score = result.get("eval_score", 0.0)
    eval_agreement = result.get("eval_agreement", False)

    verdict_html = _build_verdict_html(final_label, final_score)
    score_cards_html = _build_score_cards(
        ml_label, ml_score, llm_label, llm_score, eval_score, eval_agreement
    )

    details_md = (
        f"### Pipeline Breakdown\n\n"
        f"**ML Model:** {ml_label} with {ml_score:.0%} confidence\n"
        f"**AI Agent:** {llm_label} with {llm_score:.0%} confidence\n"
        f"**Agreement:** {'Yes' if eval_agreement else 'No'}\n"
        f"**DeepEval Score:** {eval_score:.2f} / 1.00\n"
    )

    status_md = (
        "### Processing Complete\n"
        f"- Step 1/4: Input + ingestion\n"
        f"- Step 2/4: ML inference finished ({ml_label}, {ml_score:.0%})\n"
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

            gr.Examples(
                examples=EXAMPLE_TEXTS,
                inputs=text_input,
                label="Try an example",
                examples_per_page=2,
            )

        with gr.Column(scale=6, min_width=420):
            gr.HTML('<div class="section-label">Result</div>')

            status_output = gr.Markdown(
                value="### Ready\n- Enter article text or URL, then click Analyze Article",
                elem_classes="status-panel",
            )

            verdict_output = gr.HTML()

            with gr.Row():
                with gr.Column():
                    score_cards_output = gr.HTML()

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
                    f"**Training Insight:** The ML Verdict was generated by the **{BEST_MODEL_NAME}** model, "
                    f"which proved to be the most accurate during training. See full validation metrics below."
                )
                gr.Dataframe(
                    headers=["Model", "Accuracy", "Precision", "Recall", "F1 Score"],
                    value=DASHBOARD_TABLE_DATA,
                    interactive=False
                )
                with gr.Row():
                    gr.Image(value="models/v2/plots/roc_curves.png", label="ROC Curves (Test Dataset)")
                    gr.Image(value="models/v2/plots/confusion_matrices.png", label="Confusion Matrices (Test Dataset)")

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
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_port=port, share=False, css=CUSTOM_CSS, theme=theme)
