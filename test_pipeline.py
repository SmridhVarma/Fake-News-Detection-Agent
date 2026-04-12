"""
test_pipeline.py — Full end-to-end smoke test for the inference pipeline.

Tests all nodes + all skills individually, then runs the whole chain.
"""

import os, sys, time, json
from dotenv import load_dotenv

load_dotenv()

from src.nodes.ingestion import ingestion_node
from src.nodes.llm_classifier import llm_classifier_node
from src.nodes.evaluator import evaluator_node
from src.nodes.aggregator import aggregator_node

# Skills (tested directly)
from src.utils.ingestion_tools import calculate_article_scores, fetch_article_from_url
from src.utils.analysis_tools import analyze_sentiment, check_source_credibility
from src.utils.verification_tools import cross_reference_article

# Tools (tested to confirm they load and wrap correctly)
from src.nodes.llm_classifier import ALL_TOOLS

SEP = "=" * 70

# ── Test articles ──
REAL_ARTICLE = """
WASHINGTON (Reuters) - The U.S. Federal Reserve kept interest rates unchanged on Wednesday
and signaled it still planned three rate cuts for 2024, despite recent data showing inflation
remains sticky. Fed Chair Jerome Powell said during a press conference that policymakers still
believe inflation is on a path back to the central bank's 2% target, but acknowledged the
journey may take longer than initially expected. The decision was widely anticipated by markets,
which rallied following the announcement. Treasury yields dipped, and the S&P 500 climbed 0.9%
to close at a new all-time high. Economists noted the Fed's tone remained cautiously optimistic
even as consumer prices rose 3.2% year-over-year in February, slightly above expectations.
""".strip()

FAKE_ARTICLE = """
EXPOSED: Government Officials Caught RED-HANDED Hiding Secret Alien Technology!!!
Whistle-blowers from deep inside the Pentagon have FINALLY come forward to reveal that our
government has been LYING to us for DECADES about contact with extraterrestrial beings!!!
Multiple unnamed sources confirm that a MASSIVE cover-up has been underway since the 1950s.
The mainstream media is REFUSING to cover this story because they are CONTROLLED by the very
same elites who want to keep this information from YOU, the American people!!! Share this
before they DELETE it!!! This is NOT a drill - the truth is finally coming out and THEY
cannot stop it anymore!!
""".strip()


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def test_skills():
    """Test every skill individually."""
    section("SKILL TESTS")

    # 1) calculate_features
    print("\n-- calculate_features (REAL) --")
    feat = calculate_article_scores(REAL_ARTICLE)
    for k, v in feat.items():
        print(f"  {k}: {v}")

    print("\n-- calculate_features (FAKE) --")
    feat = calculate_article_scores(FAKE_ARTICLE)
    for k, v in feat.items():
        print(f"  {k}: {v}")

    # 2) sentiment_analysis
    print("\n-- sentiment_analysis (REAL) --")
    sent = analyze_sentiment(REAL_ARTICLE)
    for k, v in sent.items():
        print(f"  {k}: {v}")

    print("\n-- sentiment_analysis (FAKE) --")
    sent = analyze_sentiment(FAKE_ARTICLE)
    for k, v in sent.items():
        print(f"  {k}: {v}")

    # 3) source_credibility
    print("\n-- source_credibility --")
    for url in ["https://www.reuters.com/article/test", "https://infowars.com/story/123", "https://unknown-blog.xyz/post"]:
        cred = check_source_credibility(url=url)
        print(f"  {cred['domain']:30s} -> tier={cred['credibility_tier']}, score={cred['credibility_score']}, known={cred['is_known_source']}")

    # 4) cross_reference (uses NewsAPI - may fail without valid key)
    print("\n-- cross_reference --")
    xref = cross_reference_article("Federal Reserve interest rates", REAL_ARTICLE[:200])
    print(f"  Related count: {xref['related_count']}")
    print(f"  Sources: {xref['sources']}")
    print(f"  Summary: {xref['corroboration_summary']}")

    # 5) fetch_url
    print("\n-- fetch_url (CNN) --")
    text = fetch_article_from_url("https://edition.cnn.com/")
    print(f"  Extracted {len(text)} characters from CNN homepage")


def test_tools():
    """Test that all LangChain tools are properly registered."""
    section("TOOL REGISTRATION CHECK")
    for t in ALL_TOOLS:
        print(f"  [OK] {t.name:30s}  -> {t.description[:80]}...")
    print(f"\n  Total tools registered: {len(ALL_TOOLS)}")


def run_pipeline_test(label, state):
    """Run the full inference chain on a state dict."""
    section(f"PIPELINE: {label}")

    # Ingestion
    t0 = time.time()
    print("\n[1/4] Ingestion ...")
    out = ingestion_node(state)
    state.update(out)
    article = state.get("article_text", "")
    print(f"  Text length  : {len(article)} chars")
    print(f"  Word count   : {state.get('word_count')}")
    print(f"  Caps ratio   : {state.get('caps_ratio')}")
    print(f"  Subjectivity : {state.get('mean_subjectivity')}")
    print(f"  Lex. density : {state.get('lexical_density')}")
    print(f"  Has dateline : {state.get('has_dateline')}")
    print(f"  Source domain: {state.get('source_domain')}")
    print(f"  [{time.time()-t0:.2f}s]")

    if len(article.strip()) < 50:
        print("  >> Article too short, skipping rest")
        return state

    # LLM Classifier
    t0 = time.time()
    print("\n[2/4] LLM Classifier ...")
    out = llm_classifier_node(state)
    state.update(out)
    print(f"  Label      : {state.get('llm_label')}")
    print(f"  Confidence : {state.get('llm_score')}")
    print(f"  Reasoning  : {state.get('llm_reasoning', '')[:120]}...")
    print(f"  [{time.time()-t0:.2f}s]")

    # DeepEval
    t0 = time.time()
    print("\n[3/4] DeepEval Evaluator ...")
    out = evaluator_node(state)
    state.update(out)
    print(f"  Eval Score : {state.get('eval_score')}")
    print(f"  [{time.time()-t0:.2f}s]")

    # Aggregator (mock ML)
    t0 = time.time()
    print("\n[4/4] Aggregator (ML mocked) ...")
    state["ml_score"] = 0.82
    state["ml_label"] = "REAL"
    out = aggregator_node(state)
    state.update(out)
    print(f"  Final Label : {state.get('final_label')}")
    print(f"  Final Score : {state.get('final_score')}")
    print(f"  Agreement   : {state.get('eval_agreement')}")
    print(f"  Summary     : {state.get('summary')}")
    print(f"  [{time.time()-t0:.2f}s]")

    return state


if __name__ == "__main__":
    overall = time.time()

    # 1. Test all skills
    test_skills()

    # 2. Test tool registration
    test_tools()

    # 3. Run pipeline with REAL article text
    run_pipeline_test("REAL article (text)", {
        "input_type": "text",
        "raw_input": REAL_ARTICLE,
    })

    # 4. Run pipeline with FAKE article text
    run_pipeline_test("FAKE article (text)", {
        "input_type": "text",
        "raw_input": FAKE_ARTICLE,
    })

    section(f"ALL DONE - Total: {time.time()-overall:.1f}s")
