# Project-Study-Classification-IA

###  Configuration System Overview

The configuration system centralizes all project settings, ensuring reproducibility and maintainability.

**Main components:**
- **Paths:** automatically resolve directories (data, models, results, etc.)
- **Constants:** control random seeds, split ratios, etc.
- **ModelConfig:** store default hyperparameters for reproducibility.
- **TextConfig:** define preprocessing rules for NLP tasks.
- **Logging:** unified format for all logs (console + file).

**Usage example:**
```python
from src.utils.config import config

print(config.paths["raw_data"])
print(config.model.random_forest["n_estimators"])

