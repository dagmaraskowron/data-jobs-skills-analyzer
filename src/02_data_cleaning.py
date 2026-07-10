import ast
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "jobs.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean_jobs.csv"


def parse_skills(skills_text):
    if pd.isna(skills_text):
        return []

    try:
        skills = ast.literal_eval(skills_text)
        if isinstance(skills, list):
            return [str(skill).lower().strip() for skill in skills]
    except (ValueError, SyntaxError):
        return []

    return []


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    df = df.dropna(subset=["job_title"])

    text_columns = [
        "seniority_level",
        "status",
        "location",
        "ownership",
        "revenue"
    ]

    for col in text_columns:
        df[col] = df[col].fillna("Unknown")

    df["post_date"] = df["post_date"].fillna("Unknown").astype(str).str.strip()

    df["job_title"] = df["job_title"].str.lower().str.strip()
    df["company"] = df["company"].str.strip()
    df["location"] = df["location"].str.strip()
    df["industry"] = df["industry"].str.strip()

    df["skills_list"] = df["skills"].apply(parse_skills)
    df["skills_clean"] = df["skills_list"].apply(lambda skills: " ".join(skills))
    df["skills_count"] = df["skills_list"].apply(len)

    df["combined_text"] = (
        df["job_title"].fillna("") + " "
        + df["seniority_level"].fillna("") + " "
        + df["industry"].fillna("") + " "
        + df["skills_clean"].fillna("")
    )

    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)

    print("Initial shape:")
    print(df.shape)

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    print("\nDuplicated rows:")
    print(df.duplicated().sum())

    clean_df = clean_data(df)

    print("\nClean shape:")
    print(clean_df.shape)

    print("\nMissing values after cleaning:")
    print(clean_df.isnull().sum())

    print("\nFirst 5 rows:")
    print(clean_df[["job_title", "company", "location", "skills_list", "skills_count"]].head())

    print("\nSkills count statistics:")
    print(clean_df["skills_count"].describe())

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"\nClean dataset saved to: {CLEAN_DATA_PATH}")