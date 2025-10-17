"""
run_text_preprocessing.py
---------------------------------------------------
Run text preprocessing pipeline on JSON dataset
and save cleaned output as CSV.
---------------------------------------------------
"""

import pandas as pd
import logging
from tqdm import tqdm
from src.data.preprocessing import TextPreprocessor
from src.utils.helpers import load_json, save_csv, timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


INPUT_PATH = "data/raw/train.json"
OUTPUT_PATH = "data/processed/train_clean.csv"


@timer
def preprocess_dataframe(df: pd.DataFrame, processor: TextPreprocessor) -> pd.DataFrame:
    """
    Apply TextPreprocessor to all text columns in the DataFrame.
    Creates new columns with the suffix '_clean'.
    """
    tqdm.pandas()

    for col in df.columns:
        logger.info(f"Preprocessing column: '{col}'...")
        df[f"{col}_clean"] = df[col].progress_apply(processor.preprocess)

    return df


if __name__ == "__main__":
    logger.info(f" Loading dataset from {INPUT_PATH}...")


    data = load_json(INPUT_PATH)
    df = pd.DataFrame(data)
    logger.info(f" Dataset loaded successfully. Shape: {df.shape}")


    processor = TextPreprocessor(
        use_stopwords=True,
        use_stemming=False
    )


    df_clean = preprocess_dataframe(df, processor)
    logger.info(" Text preprocessing completed successfully.")


    save_csv(df_clean, OUTPUT_PATH)
    logger.info(f" Cleaned dataset saved to {OUTPUT_PATH}")
