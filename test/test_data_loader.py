import json
import pytest
import pandas as pd
from src.data.data_loader import (
    _validate_file_exists, _check_integrity,
    load_training_data, load_test_data
)
@pytest.fixture
def sample_train_json(tmp_path):
    """Creates a temporary sample train.json file in JSONL format."""
    data = [
        {"text": "hello", "label": "greeting"},
        {"text": "bye", "label": "farewell"},
    ]
    file_path = tmp_path / "train.json"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path

def test_validate_file_exists(tmp_path):
    file = tmp_path / "dummy.csv"
    file.write_text("a,b,c\n1,2,3")
    _validate_file_exists(str(file))

def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        _validate_file_exists("nonexistent.csv")

def test_load_training_data(sample_train_json):
    df = load_training_data(sample_train_json)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

