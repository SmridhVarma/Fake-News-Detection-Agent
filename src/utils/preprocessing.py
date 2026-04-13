"""
preprocessing.py — Text cleaning helpers.

Includes two specific pipelines: one for traditional ML (with lemmatization)
and one for Transformer/LLMs (minimal destructive cleaning).
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.utils.preprocessing_tools import strip_publisher_patterns

# Automatically download required NLTK data on import
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Initialize the lemmatizer globally to save time
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text_for_transformers(text: str) -> str:
    """
    Minimal destructive cleaning suitable for BERT or OpenAI.
    Preserves casing, punctuation, and stopwords for context.
    """
    text = strip_publisher_patterns(text)
    text = re.sub(r"http\S+", "", text)           # strip URLs
    text = re.sub(r"<.*?>", "", text)             # strip HTML
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text.strip()


def clean_text_for_traditional_ml(text: str) -> str:
    """
    Strict normalization for TF-IDF / Random Forest. 
    Reduces vocabulary size heavily.
    """
    text = clean_text_for_transformers(text)      # base cleaning
    text = text.lower()                           # lowercase
    text = re.sub(r"[^\w\s]", "", text)           # strip punctuation/special chars
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(w) 
        for w in words 
        if w not in stop_words and len(w) > 1
    ]
    
    return " ".join(cleaned_words)


def clean_text(text: str) -> str:
    """Legacy wrapper for backward compatibility."""
    return clean_text_for_traditional_ml(text)
