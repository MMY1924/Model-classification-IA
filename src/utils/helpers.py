# Import necessary libraries for logging, timing, data handling, and system monitoring
import logging
import logging.config
# Import logging configuration from config.py
from config import LOOGGING_CONFIG
import time
from functools import wraps
import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas as pd
import psutil
import os

# Function to set up logging for the project
def setup_logging():
    """Configure and return a project-wide logger."""
    logging.config.dictConfig(LOOGGING_CONFIG)
    return logging.getLogger(__name__)

# Decorator to measure and log the execution time of functions
def timer(func):
    """Decorator to measure the execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        # Calculate elapsed time
        end_time = time.time() - sart_time  # Typo: 'sart_time' should be 'start_time'
        # Log the execution time of the function
        logging.getLogger(__name__).info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Function to validate that a DataFrame contains required columns
def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that a DataFrame contains the required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        # Log an error if columns are missing
        logging.getLogger(__name__).error(f"DataFrame is missing required columns: {missing_columns}")
        return False
    return True

# Function to plot the distribution of classes in a target variable
def plot_class_distribution(y, title="Class Distribution"):
    """Plot the distribution of classes in a target variable."""
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.show()

# Function to load data from a JSON file
def load_json(path):
    """Load a JSON file and return its contents."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(file)  # Typo: 'file' should be 'f'

# Function to save data to a JSON file
def save_json(data, path):
    """Save data to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Function to load data from a CSV file
def load_csv(path):
    # Load a CSV file into a pandas DataFrame
    return pd.read_csv(path)

# Function to save a DataFrame to a CSV file
def save_csv(df, path):
    # Save a pandas DataFrame to a CSV file
    df.to_csv(path, index=False)

# Function to get the current memory usage of the process
def get_memory_usage():
    """Return the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert bytes to MB
