import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_jobs.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "tfidf_feature_names.csv"


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    text_data = df["combined_text"].fillna("")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(text_data)

    feature_names = vectorizer.get_feature_names_out()

    print("TF-IDF matrix shape:")
    print(tfidf_matrix.shape)

    print("\nFirst 30 TF-IDF features:")
    print(feature_names[:30])

    feature_names_df = pd.DataFrame({
        "feature": feature_names
    })

    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    feature_names_df.to_csv(FEATURE_NAMES_PATH, index=False)

    print(f"\nTF-IDF matrix saved to: {TFIDF_MATRIX_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Feature names saved to: {FEATURE_NAMES_PATH}")