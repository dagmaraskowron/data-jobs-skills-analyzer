import ast
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_jobs.csv"
TFIDF_MATRIX_PATH = PROJECT_ROOT / "models" / "tfidf_matrix.joblib"
OUTPUT_PATH = PROJECT_ROOT / "data" / "jobs_with_clusters.csv"

IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

LIGHT_PINK = "#F4A7B9"
DARKER_PINK = "#E78AA0"

N_CLUSTERS = 4


def parse_skills_list(skills_text):
    try:
        skills = ast.literal_eval(skills_text)
        if isinstance(skills, list):
            return skills
    except (ValueError, SyntaxError):
        return []

    return []


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

    df["skills_list"] = df["skills_list"].apply(parse_skills_list)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    print("Cluster counts:")
    print(df["cluster"].value_counts().sort_index())

    print("\nMost common job titles by cluster:")
    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\nCluster {cluster_id}")
        print(df[df["cluster"] == cluster_id]["job_title"].value_counts().head(5))

    print("\nMost common skills by cluster:")
    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\nCluster {cluster_id}")

        cluster_skills = []

        for skills in df[df["cluster"] == cluster_id]["skills_list"]:
            cluster_skills.extend(skills)

        if len(cluster_skills) == 0:
            print("No skills listed in this cluster.")
        else:
            skill_counts = Counter(cluster_skills)
            print(pd.Series(skill_counts).sort_values(ascending=False).head(10))

    df.to_csv(OUTPUT_PATH, index=False)

    cluster_counts = df["cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=cluster_counts,
        x="Cluster",
        y="Count",
        color=LIGHT_PINK
    )
    plt.title("Liczba ofert w grupach")
    plt.xlabel("Grupa")
    plt.ylabel("Liczba ofert")
    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "07_cluster_counts.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    pca_df = pd.DataFrame({
        "PCA1": reduced_data[:, 0],
        "PCA2": reduced_data[:, 1],
        "Cluster": df["cluster"].astype(str)
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="RdPu",
        alpha=0.7
    )
    plt.title("Wizualizacja grup podobnych ofert")
    plt.xlabel("Pierwsza składowa PCA")
    plt.ylabel("Druga składowa PCA")
    plt.savefig(
        IMAGES_DIR / "08_job_clusters_pca.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
