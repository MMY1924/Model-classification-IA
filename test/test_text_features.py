import pytest
from src.features.text_features import (
    char_count, word_count, avg_word_length, TextFeatureExtractor
)
import pandas as pd

def test_char_count():
    assert char_count("abc") == 3

def test_word_count():
    assert word_count("Hello world") == 2

def test_avg_word_length():
    assert round(avg_word_length("hello world"), 2) == 5.0

def test_transform_pipeline():
    df = pd.DataFrame({"text": ["Hello world!"]})
    extractor = TextFeatureExtractor()
    df_out = extractor.transform(df, "text")
    assert "word_count" in df_out.columns
    assert df_out.shape[1] > 1
