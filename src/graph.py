"""
graph.py — LangGraph Pipeline Wiring

Imports all nodes from src/nodes/ and wires them into a StateGraph.

This is the only file that knows about the full pipeline topology.
"""

from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    ingestion_node,
    preprocess_data_node,
    train_models_node,
    evaluate_models_node,
    select_model_node,
    ml_classifier_node,
    llm_classifier_node,
    evaluator_node,
    aggregator_node,
)

# ── Build the graph ──────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("ingest", ingestion_node)
builder.add_node("preprocess_data", preprocess_data_node)
builder.add_node("train_models", train_models_node)
builder.add_node("evaluate_models", evaluate_models_node)
builder.add_node("select_model", select_model_node)
builder.add_node("ml_classify", ml_classifier_node)
builder.add_node("llm_classify", llm_classifier_node)
builder.add_node("reasoning_evaluate", evaluator_node)
builder.add_node("aggregate", aggregator_node)

builder.set_entry_point("ingest")
builder.add_edge("ingest", "preprocess_data")
builder.add_edge("preprocess_data", "train_models")
builder.add_edge("train_models", "evaluate_models")
builder.add_edge("evaluate_models", "select_model")
builder.add_edge("select_model", "ml_classify")
builder.add_edge("ml_classify", "llm_classify")
builder.add_edge("llm_classify", "reasoning_evaluate")
builder.add_edge("reasoning_evaluate", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()


# ── Public entry point ───────────────────────────────────────
def run_agent(user_input: str, input_type: str = "text") -> dict:
    """Run the full detection pipeline on an article.

    Args:
        user_input: Raw text, URL, or file path from the user.
        input_type: One of "text", "url", or "file".
    """
    initial_state: AgentState = {
        # Raw input
        "input_type": input_type,
        "raw_input": user_input,

        # Dataset / training config
        "fake_csv_path": "./data/Fake.csv",
        "true_csv_path": "./data/True.csv",
        "test_size": 0.2,
        "random_state": 42,

        # Artifact paths
        "preprocessing_artifact_path": "./models/preprocessing_artifacts.joblib",
        "training_artifact_path": "./models/training_artifacts.joblib",

        # Preprocessing / ingestion outputs
        "article_title": None,
        "article_text": "",
        "article_text_ml": "",
        "source_domain": None,
        "word_count": 0,


        # Handcrafted / style features
        "caps_ratio": None,
        "style_score": None,
        "mean_subjectivity": None,
        "lexical_density": None,
        "has_dateline": None,

        # Dataset preprocessing outputs
        "preprocessed_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "numeric_feature_cols": [],

        # Training outputs
        "model_trained": False,
        "model_path": "",
        "candidate_results": {},
        "selected_model_name": "",
        "selected_model_metrics": {},
        "saved_model_paths": {},

        # Phase 1: ML inference
        "ml_score": 0.0,
        "ml_label": "",

        # Phase 2: LLM inference
        "llm_score": 0.0,
        "llm_label": "",
        "llm_reasoning": "",

        # Evaluation
        "eval_agreement": None,
        "eval_confidence_delta": None,
        "eval_score": 0.0,

        # Final output
        "final_label": "",
        "final_score": 0.0,
        "explanation": "",
        "summary": "",
    }
    return graph.invoke(initial_state)