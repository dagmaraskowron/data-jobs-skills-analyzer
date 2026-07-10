import ast
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "jobs_with_clusters.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "job_matches.csv"


USER_SKILLS = [
    "python",
    "sql",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "machine learning",
    "r",
    "excel"
]


def parse_skills_list(skills_text):
    try:
        skills = ast.literal_eval(skills_text)
        if isinstance(skills, list):
            return [str(skill).lower().strip() for skill in skills]
    except (ValueError, SyntaxError):
        return []

    return []


def calculate_match_score(job_skills, user_skills):
    if len(job_skills) == 0:
        return 0

    job_skills_set = set(job_skills)
    user_skills_set = set(user_skills)

    matched_skills = job_skills_set.intersection(user_skills_set)
    missing_skills = job_skills_set.difference(user_skills_set)

    score = len(matched_skills) / len(job_skills_set) * 100

    return round(score, 2), sorted(matched_skills), sorted(missing_skills)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    df["skills_list"] = df["skills_list"].apply(parse_skills_list)

    match_scores = []
    matched_skills_list = []
    missing_skills_list = []

    for skills in df["skills_list"]:
        if len(skills) == 0:
            score = 0
            matched = []
            missing = []
        else:
            score, matched, missing = calculate_match_score(skills, USER_SKILLS)

        match_scores.append(score)
        matched_skills_list.append(matched)
        missing_skills_list.append(missing)

    df["match_score"] = match_scores
    df["matched_skills"] = matched_skills_list
    df["missing_skills"] = missing_skills_list

    df = df.sort_values(by="match_score", ascending=False)

    print("User skills:")
    print(USER_SKILLS)

    print("\nTop 10 best matching offers:")
    print(
        df[
            [
                "job_title",
                "company",
                "location",
                "industry",
                "skills_list",
                "match_score",
                "matched_skills",
                "missing_skills",
                "cluster"
            ]
        ].head(10)
    )

    print("\nMatch score statistics:")
    print(df["match_score"].describe())

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nJob matches saved to: {OUTPUT_PATH}")