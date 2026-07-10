import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_jobs.csv"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

LIGHT_PINK = "#F4A7B9"
DARKER_PINK = "#E78AA0"


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


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

    df["skills_list"] = df["skills_list"].apply(parse_skills_list)

    print("Dataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTop job titles:")
    print(df["job_title"].value_counts().head(10))

    print("\nTop industries:")
    print(df["industry"].value_counts().head(10))

    print("\nSkills count statistics:")
    print(df["skills_count"].describe())

    top_titles = df["job_title"].value_counts().head(10)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=top_titles.values,
        y=top_titles.index,
        color=LIGHT_PINK
    )
    plt.title("Top 10 najczęstszych stanowisk")
    plt.xlabel("Liczba ofert")
    plt.ylabel("")
    save_plot("01_top_job_titles.png")


    top_industries = df["industry"].value_counts().head(10)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=top_industries.values,
        y=top_industries.index,
        color=DARKER_PINK
    )
    plt.title("Top 10 najczęstszych branż")
    plt.xlabel("Liczba ofert")
    plt.ylabel("")
    save_plot("02_top_industries.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["skills_count"], bins=18, color=LIGHT_PINK)
    plt.title("Rozkład liczby umiejętności w ofertach")
    plt.xlabel("Liczba umiejętności")
    plt.ylabel("Liczba ofert")
    save_plot("03_skills_count_distribution.png")

    all_skills = []

    for skills in df["skills_list"]:
        all_skills.extend(skills)

    skill_counts = Counter(all_skills)
    top_skills = pd.DataFrame(
        skill_counts.most_common(15),
        columns=["Skill", "Count"]
    )

    print("\nTop skills:")
    print(top_skills)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=top_skills,
        x="Count",
        y="Skill",
        color=LIGHT_PINK
    )
    plt.title("Top 15 najczęściej wymaganych umiejętności")
    plt.xlabel("Liczba ofert")
    plt.ylabel("")
    save_plot("04_top_skills.png")

