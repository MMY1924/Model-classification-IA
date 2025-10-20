"""
run_text_features.py
---------------------
Apply text feature extraction to processed training data.
"""

import pandas as pd
import logging
from src.features.text_features import TextFeatureExtractor

logging.basicConfig(level=logging.INFO)

def main():
    input_path = "data/processed/train_no_nulls.csv"
    output_path = "data/features/train_with_text_features.csv"

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loading dataset from {input_path}")

    df = pd.read_csv(input_path)

    print("\nColumns in dataset:\n", df.columns.tolist())


    text_columns = ['question_clean', 'context_clean', 'answer_clean']


    missing_cols = [col for col in text_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing text columns: {missing_cols}")

    extractor = TextFeatureExtractor()


    for text_col in text_columns:
        logging.info(f"Extracting features for '{text_col}'...")
        df = extractor.transform(df, text_col)


        new_col_names = {
            feat: f"{text_col}_{feat}" for feat in extractor.features.keys()
        }
        df.rename(columns=new_col_names, inplace=True)


    df.to_csv(output_path, index=False)
    logging.info(f" Saved enriched dataset with all text features to {output_path}")


if __name__ == "__main__":
    main()

    