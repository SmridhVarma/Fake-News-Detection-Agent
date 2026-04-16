"""
llm_classifier_node — Phase 2: LLM-based fact-checking.

Sends article to OpenAI for analysis with a structured prompt.
"""

from src.state import AgentState
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
import os
import json
import re

from src.utils.ingestion_tools import calculate_features_tool
from src.utils.preprocessing_tools import preprocess_leakage_tool
from src.utils.analysis_tools import sentiment_analysis_tool, source_credibility_tool
from src.utils.verification_tools import cross_reference_tool

ALL_TOOLS = [
    calculate_features_tool,
    preprocess_leakage_tool,
    sentiment_analysis_tool,
    source_credibility_tool,
    cross_reference_tool,
]


def load_skill(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "skills", f"{name}.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def llm_classifier_node(state: AgentState) -> dict:
    """Uses a ReACT Agent to autonomously gather external context before classifying."""
    print("\n>>> [NODE] Starting LLM Classifier Node...")
    article_text = state.get("article_text", "")

    if not article_text or len(article_text.strip()) < 50:
        return {
            "llm_score": 0.5,
            "llm_label": "FAKE",
            "llm_reasoning": "Article content is missing or too short to analyze.",
        }

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        # Build the system prompt using our Markdown skills
        system_prompt = f"""
You are an autonomous Fact-Checking Agent. Your job is to verify the authenticity of an article using a ReACT thinking pattern.
Always outline your Thought, Action, and Observation for each step.

You have access to several tools. You MUST follow these workflows:

{load_skill("llm_classification")}

After using the tools to gather sufficient evidence, output your final classification wrapped in a JSON block exactly like this:
```json
{{
    "label": "REAL" or "FAKE",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation incorporating the tools' evidence..."
}}
```
"""
        # Create ReAct agent graph
        app = create_react_agent(llm, ALL_TOOLS)

        input_type = state.get("input_type", "text")
        raw_input = state.get("raw_input", "")

        if input_type == "url" and raw_input:
            user_prompt = f"Verify this article sourced from the URL ({raw_input}):\n\n{article_text}"
        else:
            user_prompt = f"Verify this article:\n\n{article_text}"

        messages = app.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            }
        )
        final_text = messages["messages"][-1].content

        # Parse JSON
        match = re.search(r"```json\s*(\{.*?\})\s*```", final_text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            result = {
                "llm_score": float(data.get("confidence", 0.5)),
                "llm_label": data.get("label", "FAKE").upper(),
                "llm_reasoning": data.get("reasoning", final_text),
            }
            print(">>> [NODE] Finished LLM Classifier Node.")
            return result
        else:
            result = {
                "llm_score": 0.5,
                "llm_label": "UNKNOWN",
                "llm_reasoning": "Failed to parse final JSON output. Raw output:\n"
                + final_text,
            }
            print(">>> [NODE] Finished LLM Classifier Node.")
            return result

    except Exception as e:
        print(f"Error during LLM classification: {e}")
        result = {
            "llm_score": 0.5,
            "llm_label": "FAKE",
            "llm_reasoning": f"Failed to perform LLM analysis: {str(e)}",
        }
        print(">>> [NODE] Finished LLM Classifier Node.")
        return result
