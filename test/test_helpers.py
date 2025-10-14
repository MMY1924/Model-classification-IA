# Import functions to test from helpers.py
from utils.helpers import load_json, save_json
# Import standard libraries for JSON handling and OS operations
import json, os

# Test function for load_json and save_json
# Uses pytest's tmp_path fixture to create a temporary directory for testing
# 1. Creates a test JSON file with sample data
# 2. Saves the data using save_json
# 3. Loads the data using load_json
# 4. Asserts that the loaded data matches the original data
def test_load_json(tmp_path):
    path = tmp_path / "test.json"
    data = {"key": "value"}
    save_json(data, path)
    loaded_data = load_json(path)
    assert loaded_data == data
