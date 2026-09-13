"""
Module 2 Part A: EDA, Profiling, Cleaning, and Statistical Storytelling.
Loads titanic dataset once, saves offline fallback, handles missing values per rule,
evaluates IQR outliers, tests skewness, computes 6x6 correlation, and produces 4 charts.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_PATH = ANALYTICS_DIR / "titanic.csv"


def run_eda_pipeline():
    print("==================================================")
    print("MODULE 2 - PART A: EDA & PROFILING PIPELINE")
    print("==================================================")

    # 1. Load data once & commit offline fallback
    if CSV_PATH.exists():
        print(f"Loading cached dataset from {CSV_PATH.name}...")
        df_raw = pd.read_csv(CSV_PATH)
    else:
        print("Fetching dataset from Seaborn online cache...")
        df_raw = sns.load_dataset("titanic")
        df_raw.to_csv(CSV_PATH, index=False)
        print(f"Saved committed offline fallback: {CSV_PATH.name}")

    print(f"\nDataset Shape: {df_raw.shape}")
    print("\n--- Missing Value Audit ---")
    missing_counts = df_raw.isnull().sum()
    missing_pct = (missing_counts / len(df_raw)) * 100
    missing_table = pd.DataFrame({"Missing_Count": missing_counts, "Percentage": missing_pct})
    print(missing_table[missing_table["Missing_Count"] > 0].round(2))

    # 2. Defensible Missing Handling per Threshold Rule
    # - embarked / embark_town (<5%): drop rows
    # - age (5%-30%): impute median
    # - deck (>30%): drop column due to severe sparsity (>75%)
    df = df_raw.copy()
    if "deck" in df.columns:
        df = df.drop(columns=["deck"])

    df = df.dropna(subset=["embarked", "embark_town"])
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)
    print(f"\nCleaned Shape after rule-based handling: {df.shape}")

    # 3. Univariate Outlier Analysis (IQR Rule) & Skewness
    print("\n--- Univariate Analysis: Outliers & Skewness ---")
    for col in ["age", "fare"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Column '{col}': IQR={iqr:.2f}, Outlier count={len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]
    print(f"\nFare Central Tendency: Mean={fare_mean:.2f}, Median={fare_median:.2f}, Mode={fare_mode:.2f}")
    if fare_mean > fare_median > fare_mode:
        print("Skewness Conclusion: Fare is strictly RIGHT-SKEWED (Mean > Median > Mode).")

    # 4. Bivariate Survival Breakdowns via Boolean Masks
    print("\n--- Bivariate Survival Rate Breakdowns ---")
    sr_female = df[df["sex"] == "female"]["survived"].mean()
    sr_male = df[df["sex"] == "male"]["survived"].mean()
    print(f"(a) By Sex: Female = {sr_female*100:.2f}%, Male = {sr_male*100:.2f}%")

    print("(b) By Pclass:")
    for p in sorted(df["pclass"].unique()):
        sr_p = df[df["pclass"] == p]["survived"].mean()
        print(f"    Class {p}: {sr_p*100:.2f}%")

    print("(c) By Sex & Pclass:")
    for sex in ["female", "male"]:
        for p in [1, 2, 3]:
            sr_sp = df[(df["sex"] == sex) & (df["pclass"] == p)]["survived"].mean()
            print(f"    {sex.title()} - Class {p}: {sr_sp*100:.2f}%")

    # 5. 6x6 Correlation Matrix (strictly 6 columns, excluding adult_male, alone)
    print("\n--- Correlation Heatmap (6x6) ---")
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[corr_cols].corr()

    # Find two strongest off-diagonal correlations
    corr_unstack = corr_matrix.abs().unstack()
    off_diag = corr_unstack[corr_unstack < 1.0].sort_values(ascending=False)
    top_pairs = off_diag.drop_duplicates().head(2)
    print("Top 2 strongest off-diagonal absolute correlations:")
    for (f1, f2), val in top_pairs.items():
        raw_val = corr_matrix.loc[f1, f2]
        print(f"  - {f1} <-> {f2}: r = {raw_val:.3f} (|r| = {val:.3f})")

    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("6x6 Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "eda_01_correlation_heatmap.png")
    plt.close()

    # 6. Multivariate Story Charts (4 Figures)
    print("\nGenerating 4 multivariate narrative charts...")

    # Chart 1: Survival by Sex and Passenger Class
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", palette="Set2")
    plt.title("Figure 1: Survival Rate by Pclass and Sex (Privilege & Gender Interaction)")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "eda_02_survival_pclass_sex.png")
    plt.close()

    # Chart 2: Age Distribution by Survival and Sex
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=df, x="sex", y="age", hue="survived", split=True, palette="muted")
    plt.title("Figure 2: Age Density by Gender & Survival (Child Prioritization)")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "eda_03_age_sex_survival_violin.png")
    plt.close()

    # Chart 3: Fare vs Age Scatter colored by Survival
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.7, palette={0: "red", 1: "green"})
    plt.title("Figure 3: Fare vs Age stratified by Survival")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "eda_04_fare_age_scatter.png")
    plt.close()

    # Chart 4: Family Size Interaction (sibsp + parch) vs Survival
    df_plot = df.copy()
    df_plot["family_size"] = df_plot["sibsp"] + df_plot["parch"]
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df_plot, x="family_size", y="survived", palette="Blues_d")
    plt.title("Figure 4: Survival by Family Size (Solo vs Moderate vs Large Families)")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "eda_05_family_size_survival.png")
    plt.close()

    # 7. Exploratory Sanity Check: Z-Score Standardization
    print("\n--- Exploratory Z-Score Standardization Check ---")
    age_std = (df["age"] - df["age"].mean()) / df["age"].std()
    fare_std = (df["fare"] - df["fare"].mean()) / df["fare"].std()
    print(f"Age  Standardized -> Mean: {age_std.mean():.4f}, Std: {age_std.std():.4f}")
    print(f"Fare Standardized -> Mean: {fare_std.mean():.4f}, Std: {fare_std.std():.4f}")
    print("[SUCCESS] Part A analysis and plots completed.")


if __name__ == "__main__":
    run_eda_pipeline()