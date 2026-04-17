"""
state.py — Shared Agent State

Single source of truth for the data flowing through the LangGraph pipeline.
Every node reads from and writes to this state.
"""

from typing import TypedDict, Optional, Literal, List, Dict, Any


class AgentState(TypedDict, total=False):
    """State shared across all LangGraph nodes."""

    # ── Raw input (set once at pipeline entry) ──────────────────
    input_type: Literal["text", "url", "file"]   # how the user provided the article
    raw_input: str                                # original user input (text / URL / file path)

    # ── Dataset / training configuration ───────────────────────
    fake_csv_path: str
    true_csv_path: str
    train_size: float
    val_size: float
    test_size: float
    random_state: int

    include_transformer: bool
    transformer_model_name: str
    transformer_epochs: int
    transformer_batch_size: int
    transformer_learning_rate: float

    # ── Preprocessing outputs (set by ingestion node) ──────────
    article_title: Optional[str]                  # extracted or provided title
    article_text: str                             # legacy cleaned body text
    article_text_ml: str                          # cleaned for traditional ML
    article_text_llm: str                         # cleaned for LLM / Transformers
    source_domain: Optional[str]                  # domain extracted from URL, if applicable
    word_count: int                               # token-level length of cleaned text

    # ── Handcrafted / style features (single-article inference) ─
    caps_ratio: Optional[float]
    style_score: Optional[float]
    mean_subjectivity: Optional[float]
    lexical_density: Optional[float]
    has_dateline: Optional[bool]

    # ── Dataset preprocessing outputs ──────────────────────────
    preprocessed_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    numeric_feature_cols: List[str]
    preprocessing_artifact_path: str

    # ── Intermediate training pipeline state ───────────────────
    trained_candidates_path: Optional[str]        # joblib written by train_models_node
    evaluation_artifact_path: Optional[str]       # joblib written by evaluate_models_node
    training_cache_hit: Optional[bool]            # True when v2 cache bypasses training
    candidate_model_names: Optional[List[str]]    # list of candidate model keys

    # ── Training outputs ───────────────────────────────────────
    model_trained: bool
    roc_curve_path: Optional[str]                 # path to saved ROC curve PNG
    confusion_matrix_path: Optional[str]          # path to saved confusion matrix PNG
    model_path: str
    training_artifact_path: str
    candidate_validation_results: Dict[str, Any]
    candidate_test_results: Dict[str, Any]
    candidate_results: Dict[str, Any]
    selected_model_name: str
    selected_model_validation_metrics: Dict[str, Any]
    selected_model_test_metrics: Dict[str, Any]
    selected_model_metrics: Dict[str, Any]
    saved_model_paths: Dict[str, str]

    # ── Phase 1: ML classifier ─────────────────────────────────
    ml_score: Optional[float]                     # model confidence [0-1]
    ml_label: Optional[str]                       # "FAKE" or "REAL"
    ml_model_name: Optional[str]                  # winning model key (e.g. "logistic_regression")

    # ── Phase 2: LLM classifier ────────────────────────────────
    llm_score: Optional[float]                    # LLM confidence [0-1]
    llm_label: Optional[str]                      # "FAKE" or "REAL"
    llm_reasoning: Optional[str]                  # free-text rationale from LLM
    related_articles: List[Dict[str, str]]        # NewsAPI corroborating sources (title, source, url, description)
    llm_tool_trace: List[Dict[str, Any]]          # ordered list of tool calls the ReACT agent made

    # ── Evaluation (model vs LLM comparison) ───────────────────
    eval_agreement: Optional[bool]                # True if ml_label == llm_label
    eval_confidence_delta: Optional[float]        # abs(ml_score - llm_score)
    eval_score: float                             # composite evaluation metric

    # ── Final output ───────────────────────────────────────────
    final_label: str                              # "FAKE" or "REAL"
    final_score: float                            # aggregated confidence
    explanation: str                              # human-readable explanation
    summary: str                                  # one-line verdict for UI display
    ml_weight: Optional[float]                    # weight applied to ML signal in final aggregation
    llm_weight: Optional[float]                   # weight applied to LLM signal in final aggregation