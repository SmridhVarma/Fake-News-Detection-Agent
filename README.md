# Fake News Detection Agent

LangGraph-based agentic pipeline for detecting fake news using **two-phase classification** (ML + LLM) with **DeepEval** for LLM evaluation.

## Architecture

```
Article Input (Gradio UI)
        │
        ▼
┌─────────────────┐
│  Ingestion Node  │  Preprocess article text
└────────┬────────┘
         ▼
┌─────────────────┐
│  Training Node   │  Train / load ML model (via skills/)
└────────┬────────┘
         ▼
┌─────────────────┐
│  ML Classifier   │  Phase 1: ML model inference → score
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM Classifier  │  Phase 2: OpenAI fact-check → score + reasoning
└────────┬────────┘
         ▼
┌─────────────────┐
│  Evaluator Node  │  DeepEval metrics on LLM output
└────────┬────────┘
         ▼
┌─────────────────┐
│  Aggregator Node │  Combine scores → final verdict + summary
└────────┬────────┘
         ▼
   Final Output (Gradio UI)
```

## Project Structure

```
├── main.py                    # Gradio UI entry point
├── src/
│   ├── state.py               # AgentState definition (shared state)
│   ├── graph.py               # LangGraph wiring (imports nodes, builds graph)
│   ├── nodes/                 # ← Each pipeline step is a node
│   │   ├── ingestion.py       #    Text preprocessing
│   │   ├── training.py        #    ML model training / loading
│   │   ├── ml_classifier.py   #    Phase 1: ML inference
│   │   ├── llm_classifier.py  #    Phase 2: LLM fact-check
│   │   ├── evaluator.py       #    DeepEval evaluation
│   │   └── aggregator.py      #    Final score aggregation
│   └── utils/                 # ← Helper functions (called by nodes)
│       ├── preprocessing.py   #    Text cleaning
│       ├── model_io.py        #    Save / load models
│       └── prompts.py         #    LLM prompt templates
├── skills/                    # Dedicated reusable skill modules
├── models/                    # Saved trained ML models
├── data/                      # Training / evaluation datasets
├── .env.example               # Environment variable template
├── .gitignore
└── requirements.txt
```

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy env template and add your keys
cp .env.example .env

# 5. Run the app
python main.py
```

### Data Source
```
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download
```