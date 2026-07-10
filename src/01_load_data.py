import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "jobs.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    print("Looking for file here:")
    print(DATA_PATH)

    print("\nFile exists?")
    print(DATA_PATH.exists())

    df = load_data()

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nColumns:")
    print(df.columns.tolist())