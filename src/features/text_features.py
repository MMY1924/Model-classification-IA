import re
import string
import logging
from typing import Dict, Any
from nltk.tokenize import sent_tokenize, word_tokenize
from src.utils.helpers import timer
import pandas as pd

logger = logging.getLogger(__name__)

def char_count(text: str) -> int:
    """Count the total number of characters in a text"""
    return len(text)

def word_count(text: str) -> int:
    """Count the number of words in a text"""
    return len(word_tokenize(text))

def sentence_count(text: str) -> int:
    """Count the number of sentences in a text"""
    return len(sent_tokenize(text))

def punctuation_count(text: str) -> int:
    """Count the number of punctuation marks in a text."""
    return sum(1 for c in text if c in string.punctuation)

def avg_word_length(text: str) -> float:
    """"Count the average length of words in a text"""
    if not isinstance(text, str):
        return 0.0
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0

def uppercase_ratio(text: str) -> int:
    """Calculate the ratio of uppercase letters to total letters"""
    letters = [c for c in text if c.isalpha()]
    upper = [c for c in text if c.isupper()]
    return len(upper) / len(letters) if letters else 0

def special_char_ratio(text: str) -> float:
    """Calculate the ratio of special characters
    (not alphanumeric or characters)"""
    total_chars = len(text)
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    return special_chars / total_chars if total_chars > 0 else 0

def digit_ratio(text: str) -> float:
    """Calculate the ratio of digits to total characters"""
    total_chars = len(text)
    digits = sum(c.isdigit() for c in text)
    return digits / total_chars if total_chars > 0 else 0


class TextFeatureExtractor:
    """
    Pipeline to extract structural text-based features for ML models.

    Attributes:
        features (list): List of functions to apply for feature extraction.
    """

    def __init__(self):
        self.features = {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "punctuation_count": punctuation_count,
            "uppercase_ratio": uppercase_ratio,
            "special_char_ratio": special_char_ratio,

        }

    @timer
    def transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Apply all feature extraction functions to a text column.

        Args:
            df (pd.DataFrame): Input DataFrame containing text data.
            text_col (str): Name of the column containing text.

        Returns:
            pd.DataFrame: Original DataFrame with new feature columns appended.
        """
        for feat_name, func in self.features.items():
            logger.info(f"Extracting feature: {feat_name}")
            df[feat_name] = df[text_col].apply(func)
        return df

