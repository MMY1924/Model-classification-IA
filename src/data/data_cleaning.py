import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def main():
    data_path = Path("data/processed/train_clean.csv")
    output_path = Path("data/processed/train_no_nulls.csv")

    logging.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    logging.info(f"Initial shape: {df.shape}")

    # Columnas críticas para mantener datos válidos
    critical_cols = ["question_clean", "type_clean", "answer_clean", "context_clean"]

    # Eliminar filas con nulos en columnas importantes
    df_clean = df.dropna(subset=critical_cols)

    # Reemplazar posibles no-string por string vacío o str conversion
    for col in critical_cols:
        df_clean[col] = df_clean[col].astype(str)

    logging.info(f"Final shape after cleaning: {df_clean.shape}")

    # Guardar dataset limpio
    df_clean.to_csv(output_path, index=False)
    logging.info(f"Cleaned dataset saved at {output_path}")

if __name__ == "__main__":
    main()
