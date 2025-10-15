"""
data_loader.py
Utility functions to load, validate, and check data
integrity for the project.
"""

import os
import json
import pandas as pd
from typing import Tuple, Dict, Any
from src.utils.config import Config

config = Config()
# Check if a file exists at the given path
# Raises FileNotFoundError if the file does not exist
def _validate_file_exists(filepath: str ) -> None:
    """Check if a file exists at the given path."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

# Validate that a DataFrame contains all expected columns
# Raises ValueError if columns are missing
def _validate_schema(df: pd.DataFrame, expected_columns: list) -> None:
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

# Check that DataFrame columns have the expected data types
# Raises TypeError if any column has an incorrect dtype
def _check_dtypes(df: pd.DataFrame, expected_dtypes: Dict[str, Any]) -> None:
    for col, dtype in expected_dtypes.items():
        if col in df.columns and not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
            raise TypeError(f"Column '{col}' has incorrect dtype: expected {dtype}, got {df[col].dtype}")

# Check for missing values and duplicate rows in a DataFrame
# Raises ValueError if any are found
def _check_integrity(df: pd.DataFrame) -> None:
    if df.isnull().sum().sum() > 0:
        raise ValueError("DataFrame contains missing values")
    if df.duplicated().sum() > 0:
        raise ValueError("DataFrame contains duplicate rows")

# Load and validate training data from a JSON file
# Checks file existence, loads data, and validates integrity
def load_training_data(path):
    """Load and validate training data from a JSON file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


# Load and validate test data from a JSON file
# Checks file existence, loads data, and validates integrity
def load_test_data(path: str = os.path.join(config.paths['raw_data'], 'test.json')) -> pd.DataFrame:
    """Load and validate test data from a JSON file."""
    _validate_file_exists(path)
    df = pd.read_json(path, lines=True)
    _check_integrity(df)
    return df

# Load and validate processed data from a CSV file
# Checks file existence and loads data
def load_processed_data(filename: str = 'processed_data.csv') -> pd.DataFrame:
    """Load and validate processed data from a CSV file."""
    path = os.path.join(config.paths['processed_data'], filename)
    _validate_file_exists(path)
    df = pd.read_csv(path)
    _check_integrity(df)
    return df
