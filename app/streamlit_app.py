import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

JOBS_PATH = PROJECT_ROOT / "data" / "job_matches.csv"
SKILLS_PATH = PROJECT_ROOT / "data" / "skills_summary.csv"

LIGHT_PINK = "#F4A7B9"


DEFAULT_USER_SKILLS = [
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


def parse_list(text):
    try:
        values = ast.literal_eval(text)
        if isinstance(values, list):
            return values
    except (ValueError, SyntaxError):
        return []

    return []


def calculate_match_score(job_skills, user_skills):
    if len(job_skills) == 0:
        return 0, [], []

    job_skills_set = set(job_skills)
    user_skills_set = set(user_skills)

    matched_skills = sorted(job_skills_set.intersection(user_skills_set))
    missing_skills = sorted(job_skills_set.difference(user_skills_set))

    score = len(matched_skills) / len(job_skills_set) * 100

    return round(score, 2), matched_skills, missing_skills


def plot_horizontal_bar_chart(data, x_col, y_col, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.barh(data[y_col], data[x_col], color=LIGHT_PINK)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)


@st.cache_data
def load_data():
    jobs = pd.read_csv(JOBS_PATH)
    skills = pd.read_csv(SKILLS_PATH)

    jobs["skills_list"] = jobs["skills_list"].apply(parse_list)

    return jobs, skills


st.set_page_config(
    page_title="Data Jobs Skills Analyzer",
    layout="wide"
)


st.title("Data Jobs Skills Analyzer")

st.write(
    "Aplikacja analizuje oferty pracy z obszaru data science i pokrewnych stanowisk. "
    "Pozwala sprawdzić najczęściej wymagane umiejętności, grupy podobnych ofert "
    "oraz dopasowanie ofert do wybranego profilu umiejętności."
)

st.info(
    "Projekt wykorzystuje analizę tekstu, TF-IDF, clustering oraz prosty system dopasowania ofert."
)


jobs_df, skills_df = load_data()


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Podsumowanie",
        "Umiejętności",
        "Grupy podobnych ofert",
        "Dopasowanie ofert"
    ]
)


with tab1:
    st.header("Podsumowanie danych")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Liczba ofert", len(jobs_df))

    with col2:
        st.metric("Liczba firm", jobs_df["company"].nunique())

    with col3:
        st.metric("Średnia liczba umiejętności", round(jobs_df["skills_count"].mean(), 2))

    st.subheader("Najczęstsze stanowiska")

    top_titles = (
        jobs_df["job_title"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_titles.columns = ["Stanowisko", "Liczba ofert"]

    plot_horizontal_bar_chart(
        data=top_titles,
        x_col="Liczba ofert",
        y_col="Stanowisko",
        title="Najczęstsze stanowiska",
        xlabel="Liczba ofert"
    )

    st.subheader("Najczęstsze branże")

    top_industries = (
        jobs_df["industry"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_industries.columns = ["Branża", "Liczba ofert"]

    plot_horizontal_bar_chart(
        data=top_industries,
        x_col="Liczba ofert",
        y_col="Branża",
        title="Najczęstsze branże",
        xlabel="Liczba ofert"
    )

    st.info(
        "Większość ofert w zbiorze dotyczy stanowiska Data Scientist, "
        "dlatego wyniki należy interpretować głównie w kontekście ofert data science."
    )


with tab2:
    st.header("Analiza umiejętności")

    st.subheader("Top 20 najczęściej wymaganych umiejętności")

    top_skills = (
        skills_df
        .sort_values(by="Count", ascending=False)
        .head(20)
    )

    plot_horizontal_bar_chart(
        data=top_skills,
        x_col="Count",
        y_col="Skill",
        title="Top 20 najczęściej wymaganych umiejętności",
        xlabel="Liczba ofert"
    )

    st.subheader("Kategorie umiejętności")

    category_summary = (
        skills_df
        .groupby("Category")["Count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    plot_horizontal_bar_chart(
        data=category_summary,
        x_col="Count",
        y_col="Category",
        title="Wymagane umiejętności według kategorii",
        xlabel="Liczba wystąpień"
    )

    st.subheader("Tabela umiejętności")

    st.dataframe(
        skills_df.sort_values(by="Count", ascending=False),
        use_container_width=True
    )


with tab3:
    st.header("Grupy podobnych ofert")

    st.write(
        "Oferty zostały pogrupowane na podstawie podobieństwa tekstowego. "
        "Do zamiany tekstu na liczby wykorzystano TF-IDF, a do grupowania algorytm KMeans."
    )

    cluster_counts = (
        jobs_df["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_counts.columns = ["Grupa", "Liczba ofert"]
    cluster_counts["Grupa"] = cluster_counts["Grupa"].astype(str)

    st.subheader("Liczba ofert w grupach")

    plot_horizontal_bar_chart(
        data=cluster_counts,
        x_col="Liczba ofert",
        y_col="Grupa",
        title="Liczba ofert w grupach",
        xlabel="Liczba ofert"
    )

    selected_cluster = st.selectbox(
        "Wybierz grupę ofert:",
        sorted(jobs_df["cluster"].unique())
    )

    cluster_df = jobs_df[jobs_df["cluster"] == selected_cluster]

    st.write(f"Liczba ofert w wybranej grupie: **{len(cluster_df)}**")

    st.subheader("Najczęstsze stanowiska w wybranej grupie")
    st.write(cluster_df["job_title"].value_counts().head(10))

    st.subheader("Najczęstsze umiejętności w wybranej grupie")

    cluster_skills = []

    for skills in cluster_df["skills_list"]:
        cluster_skills.extend(skills)

    if len(cluster_skills) == 0:
        st.warning("W tej grupie oferty nie mają podanych umiejętności.")
    else:
        cluster_skills_df = (
            pd.Series(cluster_skills)
            .value_counts()
            .head(15)
            .reset_index()
        )
        cluster_skills_df.columns = ["Skill", "Count"]

        plot_horizontal_bar_chart(
            data=cluster_skills_df,
            x_col="Count",
            y_col="Skill",
            title="Najczęstsze umiejętności w wybranej grupie",
            xlabel="Liczba ofert"
        )

    st.subheader("Przykładowe oferty z wybranej grupy")

    st.dataframe(
        cluster_df[
            [
                "job_title",
                "company",
                "location",
                "industry",
                "skills_clean",
                "match_score"
            ]
        ].head(20),
        use_container_width=True
    )


with tab4:
    st.header("Dopasowanie ofert do profilu")

    st.write(
        "Wybierz swoje umiejętności, a aplikacja pokaże oferty najlepiej dopasowane "
        "do wybranego profilu."
    )

    all_skills = sorted(skills_df["Skill"].unique())

    selected_skills = st.multiselect(
        "Wybierz swoje umiejętności:",
        all_skills,
        default=[skill for skill in DEFAULT_USER_SKILLS if skill in all_skills]
    )

    min_score = st.slider(
        "Minimalny match score:",
        min_value=0,
        max_value=100,
        value=50,
        step=5
    )

    results = jobs_df.copy()

    scores = []
    matched_all = []
    missing_all = []

    for skills in results["skills_list"]:
        score, matched, missing = calculate_match_score(skills, selected_skills)
        scores.append(score)
        matched_all.append(", ".join(matched))
        missing_all.append(", ".join(missing))

    results["match_score"] = scores
    results["matched_skills"] = matched_all
    results["missing_skills"] = missing_all

    results = results[results["match_score"] >= min_score]
    results = results.sort_values(by="match_score", ascending=False)

    st.subheader("Najlepiej dopasowane oferty")

    st.write(f"Liczba ofert spełniających warunek: **{len(results)}**")

    st.dataframe(
        results[
            [
                "job_title",
                "company",
                "location",
                "industry",
                "salary",
                "match_score",
                "matched_skills",
                "missing_skills",
                "cluster"
            ]
        ].head(30),
        use_container_width=True
    )

    st.info(
        "Match score oznacza procent wymaganych umiejętności z oferty, "
        "które znajdują się w wybranym profilu użytkownika."
    )