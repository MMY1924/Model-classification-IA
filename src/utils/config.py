"""
config.py
Centralized configuration system for the project.
All paths, constants, and hyperparameters are defined here.
"""

import os
from dataclasses import dataclass, field

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory paths for data, models, results, and reports
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
# FEATURES_DIR is missing in the original code, add it for completeness
FEATURES_DIR = os.path.join(DATA_DIR, 'features')

# Random seed for reproducibility
RANDOM_SEED = 42
# Proportion of data to use for testing
TEST_SIZE = 0.2
# Proportion of data to use for validation
VALIDATION_SIZE = 0.1

@dataclass
class ModelConfig:
    """Default configuration for model training."""
    # Hyperparameters for Random Forest model
    random_forest: dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'random_state': RANDOM_SEED
    })
    # Hyperparameters for Logistic Regression model
    logistic_regression: dict = field(default_factory=lambda: {
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 300,
        "multi_class": "multinomial",
        "random_state": RANDOM_SEED
    })

@dataclass
class TextConfig:
    """Configuration for text processing."""
    # Whether to convert text to lowercase
    lowercase: bool = True
    # Whether to remove punctuation from text
    remove_punctuation: bool = True
    # Minimum word length to keep
    min_word_length: int = 3
    # Language for text processing
    lenguage: str = 'english'

# Logging configuration for the project
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": os.path.join(BASE_DIR, "results", "project.log"),
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

@dataclass
class Config:
    """Main configuration class to access all project settings."""
    # Dictionary of important paths
    paths: dict = field(default_factory=lambda: {
        "base": BASE_DIR,
        "raw_data": RAW_DATA_DIR,
        "processed_data": PROCESSED_DATA_DIR,
        "features": FEATURES_DIR,
        "models": MODELS_DIR,
        "results": RESULTS_DIR,
        "reports": REPORTS_DIR,
    })
    # Dictionary of constants used throughout the project
    constants: dict = field(default_factory=lambda: {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
    })
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    # Text processing configuration
    text: TextConfig = field(default_factory=TextConfig)
    # Logging configuration
    logging: dict = field(default_factory=lambda: LOGGING_CONFIG)

# Global config object to be used throughout the project
config = Config()
