"""
config.py
Centralized configuration system for the project.
All paths, constants, and hyperparameters are defined here.
"""

import os
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

@dataclass
class ModelConfig:
    """Defult configuration for model training."""
    random_forest: dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        random_state: RANDOM_SEED
    })
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
    lowercase: bool = True
    remove_punctuation: bool = True
    min_word_length: int = 3
    lenguage: str = 'english'

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
    paths: dict = field(default_factory=lambda: {
        "base": BASE_DIR,
        "raw_data": RAW_DATA_DIR,
        "processed_data": PROCESSED_DATA_DIR,
        "features": FEATURES_DIR,
        "models": MODELS_DIR,
        "results": RESULTS_DIR,
        "reports": REPORTS_DIR,
    })
    constants: dict = field(default_factory=lambda: {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
    })
    model: ModelConfig = field(default_factory=ModelConfig)
    text: TextConfig = field(default_factory=TextConfig)
    logging: dict = field(default_factory=lambda: LOGGING_CONFIG)

config = Config()





