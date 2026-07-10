import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_jobs.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "skills_summary.csv"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

LIGHT_PINK = "#F4A7B9"
DARKER_PINK = "#E78AA0"


SKILL_CATEGORIES = {
    "Programming": [
        "python", "r", "sql", "scala", "java", "c++", "javascript"
    ],
    "Machine Learning": [
        "machine learning", "deep learning", "scikit-learn",
        "tensorflow", "pytorch", "nlp", "computer vision"
    ],
    "Cloud": [
        "aws", "azure", "gcp"
    ],
    "Data Engineering": [
        "spark", "hadoop", "database", "etl", "airflow", "kafka"
    ],
    "Visualization": [
        "tableau", "power bi", "matplotlib", "seaborn"
    ],
    "Tools": [
        "git", "docker", "linux", "excel"
    ]
}


def parse_skills_list(skills_text):

    try:
        skills = ast.literal_eval(skills_text)
        if isinstance(skills, list):
            return skills
    except (ValueError, SyntaxError):
        return []

    return []


def assign_skill_category(skill: str) -> str:

    for category, skills in SKILL_CATEGORIES.items():
        if skill in skills:
            return category

    return "Other"


def save_plot(filename: str) -> None:

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df["skills_list"] = df["skills_list"].apply(parse_skills_list)

    all_skills = []

    for skills in df["skills_list"]:
        all_skills.extend(skills)

    skill_counts = Counter(all_skills)

    skills_summary = pd.DataFrame(
        skill_counts.items(),
        columns=["Skill", "Count"]
    ).sort_values(by="Count", ascending=False)

    skills_summary["Category"] = skills_summary["Skill"].apply(assign_skill_category)
    skills_summary["Percentage"] = skills_summary["Count"] / len(df) * 100

    print("Top 20 skills:")
    print(skills_summary.head(20))

    print("\nSkill categories:")
    print(skills_summary.groupby("Category")["Count"].sum().sort_values(ascending=False))

    skills_summary.to_csv(OUTPUT_PATH, index=False)


    top_skills = skills_summary.head(20)

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=top_skills,
        x="Count",
        y="Skill",
        color=LIGHT_PINK
    )
    plt.title("Wymagane umiejętności według kategorii")
    plt.xlabel("Liczba wystąpień")
    plt.ylabel("")
    save_plot("05_top_20_skills.png")

    category_counts = (
        skills_summary
        .groupby("Category")["Count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=category_counts,
        x="Count",
        y="Category",
        color=DARKER_PINK
    )
    plt.title("Wymagane umiejętności według kategorii")
    plt.xlabel("Liczba wystąpień")
    plt.ylabel("")
    save_plot("06_skill_categories.png")

