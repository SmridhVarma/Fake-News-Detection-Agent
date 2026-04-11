"""
state.py — Shared Agent State

Single source of truth for the data flowing through the LangGraph pipeline.
Every node reads from and writes to this state.
"""

from typing import TypedDict, Optional, Literal


class AgentState(TypedDict):
    """State shared across all LangGraph nodes."""

    # ── Raw input (set once at pipeline entry) ──────────────────
    input_type: Literal["text", "url", "file"]  # how the user provided the article
    raw_input: str                               # original user input (text / URL / file path)

    # ── Preprocessing outputs (set by ingestion node) ──────────
    article_title: Optional[str]                 # extracted or provided title
    article_text: str                            # legacy cleaned body text
    article_text_ml: str                         # cleaned for traditional ML (lemmatized, lowercased)
    article_text_llm: str                        # cleaned for LLM / Transformers (preserves context)
    source_domain: Optional[str]                 # domain extracted from URL, if applicable
    word_count: int                              # token-level length of cleaned text

    # ── Style signals (set by style_check) ─────────────────────
    caps_ratio: Optional[float]                  # fraction of ALL-CAPS words
    #exclamation_count: Optional[int]             # number of '!' in text
    #amplifier_word_count: Optional[int]          # count of sensationalist words
    style_score: Optional[float]                 # composite style suspicion score [0-1]
    mean_subjectivity: Optional[float]
    lexical_density: Optional[float]
    has_dateline: Optional[bool]

    # ── Source credibility (set by source_score) ───────────────
    #source_credibility: Optional[float]          # domain credibility lookup score [0-1]

    # ── Phase 1: ML classifier ─────────────────────────────────
    ml_score: Optional[float]=None                              # model confidence [0-1]
    ml_label: Optional[str]=None                                # "FAKE" or "REAL"

    # ── Phase 2: LLM classifier ────────────────────────────────
    llm_score: Optional[float]=None                             # LLM confidence [0-1]
    llm_label: Optional[str]=None                               # "FAKE" or "REAL"
    llm_reasoning: Optional[str]=None                           # free-text rationale from LLM

    # ── Evaluation (model vs LLM comparison) ───────────────────
    eval_agreement: Optional[bool]               # True if ml_label == llm_label
    eval_confidence_delta: Optional[float]       # abs(ml_score - llm_score)
    eval_score: float                            # composite evaluation metric

    # ── Final output ───────────────────────────────────────────
    final_label: str                             # "FAKE" or "REAL"
    final_score: float                           # aggregated confidence
    explanation: str                             # Phase 2: human-readable explanation
    summary: str                                 # one-line verdict for UI display

    # ── Flags ──────────────────────────────────────────────────
    model_trained: bool
