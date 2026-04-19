"""
eda.py
------
Exploratory Data Analysis module.
Generates all charts and statistical summaries for understanding the employee dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]


def setup_output_dir():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("images", exist_ok=True)


def plot_performance_distribution(df):
    """Bar + Pie chart for target class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    counts = df["PerformanceLabel"].value_counts()
    
    # Bar chart
    axes[0].bar(counts.index, counts.values, color=["#C44E52", "#4C72B0", "#55A868"])
    axes[0].set_title("Employee Performance Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Performance Category")
    axes[0].set_ylabel("Number of Employees")
    for i, (idx, val) in enumerate(counts.items()):
        axes[0].text(i, val + 5, str(val), ha="center", fontweight="bold")
    
    # Pie chart
    axes[1].pie(counts.values, labels=counts.index,
                colors=["#C44E52", "#4C72B0", "#55A868"],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Performance Category %", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("images/01_performance_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/01_performance_distribution.png")


def plot_department_performance(df):
    """Performance breakdown by department."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dept_perf = df.groupby(["Department", "PerformanceLabel"]).size().unstack(fill_value=0)
    dept_perf_pct = dept_perf.div(dept_perf.sum(axis=1), axis=0) * 100
    
    dept_perf_pct.plot(kind="bar", stacked=True, ax=ax,
                       color=["#C44E52", "#4C72B0", "#55A868"])
    ax.set_title("Performance Distribution by Department", fontsize=14, fontweight="bold")
    ax.set_xlabel("Department")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Performance", loc="upper right")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("images/02_department_performance.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/02_department_performance.png")


def plot_age_salary_distribution(df):
    """Age and salary histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(df["Age"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Age Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df["Age"].mean(), color="red", linestyle="--",
                    label=f"Mean: {df['Age'].mean():.1f}")
    axes[0].legend()
    
    axes[1].hist(df["Salary"], bins=30, color="#55A868", edgecolor="white")
    axes[1].set_title("Salary Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Salary ($)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(df["Salary"].mean(), color="red", linestyle="--",
                    label=f"Mean: ${df['Salary'].mean():,.0f}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("images/03_age_salary_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/03_age_salary_distribution.png")


def plot_training_vs_performance(df):
    """Training hours vs performance - box plots."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    order = ["Low", "Medium", "High"]
    palette = {"Low": "#C44E52", "Medium": "#4C72B0", "High": "#55A868"}
    
    sns.boxplot(data=df, x="PerformanceLabel", y="TrainingHours",
                order=order, palette=palette, ax=ax)
    ax.set_title("Training Hours by Performance Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance Level")
    ax.set_ylabel("Training Hours per Year")
    
    plt.tight_layout()
    plt.savefig("images/04_training_vs_performance.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/04_training_vs_performance.png")


def plot_satisfaction_heatmap(df):
    """Correlation heatmap of key features."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Select key features for readability
    key_features = [
        "TrainingHours", "ProjectsCompleted", "JobSatisfaction",
        "WorkLifeBalance", "ManagerRating", "Absences",
        "OvertimeHours", "PromotionsLast5Yr", "LastReviewScore",
        "PerformanceScore"
    ]
    key_features = [f for f in key_features if f in numeric_df.columns]
    corr_matrix = numeric_df[key_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("images/05_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/05_correlation_heatmap.png")


def plot_job_level_salary(df):
    """Salary distribution by job level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    level_order = ["Junior", "Mid", "Senior", "Lead", "Manager"]
    
    sns.violinplot(data=df, x="JobLevel", y="Salary",
                   order=level_order, palette="Blues", ax=ax)
    ax.set_title("Salary Distribution by Job Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Job Level")
    ax.set_ylabel("Salary ($)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    
    plt.tight_layout()
    plt.savefig("images/06_salary_by_level.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/06_salary_by_level.png")


def plot_absences_impact(df):
    """Scatter plot: absences vs last review score colored by performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_map = {"Low": "#C44E52", "Medium": "#4C72B0", "High": "#55A868"}
    
    for label, group in df.groupby("PerformanceLabel"):
        ax.scatter(group["Absences"], group["LastReviewScore"],
                   c=color_map[label], label=label, alpha=0.6, s=40)
    
    ax.set_title("Absences vs Review Score (by Performance)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Absences per Year")
    ax.set_ylabel("Last Review Score (1-10)")
    ax.legend(title="Performance")
    
    plt.tight_layout()
    plt.savefig("images/07_absences_vs_review.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/07_absences_vs_review.png")


def generate_summary_stats(df):
    """Save summary statistics to CSV."""
    summary = df.describe().round(2)
    summary.to_csv("outputs/summary_statistics.csv")
    print("Saved: outputs/summary_statistics.csv")
    
    # Value counts for categoricals
    for col in ["PerformanceLabel", "Department", "JobLevel", "Gender"]:
        if col in df.columns:
            df[col].value_counts().to_csv(f"outputs/{col.lower()}_distribution.csv")
    
    print("Distribution CSVs saved to outputs/")


def run_eda(filepath="data/employee_data.csv"):
    """Run the complete EDA pipeline."""
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS STARTED")
    print("=" * 50)
    
    setup_output_dir()
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nBasic Stats:\n{df.describe().round(2)}")
    
    print("\nGenerating visualizations...")
    plot_performance_distribution(df)
    plot_department_performance(df)
    plot_age_salary_distribution(df)
    plot_training_vs_performance(df)
    plot_satisfaction_heatmap(df)
    plot_job_level_salary(df)
    plot_absences_impact(df)
    generate_summary_stats(df)
    
    print("\n" + "=" * 50)
    print("EDA COMPLETE - All charts saved to images/")
    print("=" * 50)
    
    return df


if __name__ == "__main__":
    run_eda()
