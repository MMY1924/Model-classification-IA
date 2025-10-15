import pytest
from src.data.preprocessing import (
    clean_text, remove_special_characters, remove_html,
    tokenize_text, remove_stopwords, lemmatize_tokens, TextPreprocessor
)

def test_clean_text_basic():
    assert clean_text("Hello <b>World!</b>") == "hello world"

def test_remove_special_characters():
    assert remove_special_characters("Hello@123") == "Hello"

def test_tokenize_text():
    tokens = tokenize_text("This is a test")
    assert "test" in tokens

def test_pipeline():
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess("This is a <b>TEST</b>!!!")
    assert isinstance(result, str)
    assert "test" in result
