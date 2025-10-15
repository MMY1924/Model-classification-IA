import re
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from src.utils.helpers import timer


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
logger = logging.getLogger(__name__)

@timer
def clean_text(text: str) -> str:
    """"Full cleaning pipeline for text data."""
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    return text

def remove_special_characters(text: str) -> str:
    """Remove special characters and punctuation from a text."""
    return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_html(text: str) -> str:
    """Remove HTML tags from a text."""
    return BeautifulSoup(text, "html.parser").get_text()

def remove_urls(text: str) -> str:
    """Remove URLs from a text."""
    return re.sub(r'http\S+|www\S+', '', text)

def remove_extra_whitespace(text: str) -> str:
    """Normalize whitespace in a text."""
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_text(text: str):
    """Split text into tokens(words)."""
    return word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords from a list of tokens."""
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize a list of tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]

def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply stemming (reduce to a root form)."""
    return [stemmer.stem(word) for token in tokens]


class TextPreprocessor:
    """Pipeline for text preprocessing."""

    def __init__(self, use_stopwords=True, use_stemming=False):
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming

    @timer
    def preprocess(self, text: str) -> str:
        """Apply all text cleaning and normalization steps."""
        text = clean_text(text)
        tokens = tokenize_text(text)
        if self.use_stopwords:
            tokens = remove_stopwords(tokens)
        if self.use_stemming:
            tokens = stem_tokens(tokens)
        else:
            tokens = lemmatize_tokens(tokens)
        return ' '.join(tokens)
