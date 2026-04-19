"""
streamlit_app.py
----------------
Interactive browser dashboard for Employee Performance Predictor.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import sys

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Make src imports work ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4C72B0;
        margin-bottom: 0.5rem;
    }
    .badge-high   { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.85rem; font-weight:600; }
    .badge-medium { background:#cce5ff; color:#004085; padding:3px 10px; border-radius:12px; font-size:0.85rem; font-weight:600; }
    .badge-low    { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:12px; font-size:0.85rem; font-weight:600; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #4C72B0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper: load everything ───────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    path = "data/employee_data.csv"
    if not os.path.exists(path):
        from src.generate_data import generate_employee_data
        os.makedirs("data", exist_ok=True)
        df = generate_employee_data(1000)
        df.to_csv(path, index=False)
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    try:
        with open("models/best_model.pkl",    "rb") as f: model  = pickle.load(f)
        with open("models/scaler.pkl",         "rb") as f: scaler = pickle.load(f)
        with open("models/feature_names.pkl",  "rb") as f: feats  = pickle.load(f)
        with open("models/model_metadata.json","r")  as f: meta   = json.load(f)
        return model, scaler, feats, meta
    except FileNotFoundError:
        return None, None, None, None

# ── Prediction logic (mirrors src/predict.py) ─────────────────────────────────
def predict_single(emp, model, scaler, feature_names):
    e = dict(emp)
    e["SalaryPerExperience"]  = e["Salary"]          / (e["TotalExperience"] + 1)
    e["TrainingEfficiency"]   = e["ProjectsCompleted"]/ (e["TrainingHours"]   + 1)
    e["OverallSatisfaction"]  = (e["JobSatisfaction"] + e["WorkLifeBalance"] + e["ManagerRating"]) / 3
    e["AbsenceRate"]          = e["Absences"] / 365 * 100
    e["CompanyLoyalty"]       = e["YearsAtCompany"]   / (e["TotalExperience"] + 1)
    e["PromotionVelocity"]    = e["PromotionsLast5Yr"]/ (e["YearsAtCompany"]  + 1)
    e["JobLevel_Encoded"]     = {"Junior":1,"Mid":2,"Senior":3,"Lead":4,"Manager":5}.get(e.get("JobLevel","Mid"), 2)
    e["Gender_Encoded"]       = {"Male":0,"Female":1,"Other":2}.get(e.get("Gender","Male"), 0)
    for dept in ["Sales","IT","HR","Finance","Marketing","Operations","R&D"]:
        e[f"Dept_{dept}"] = 1 if e.get("Department") == dept else 0
    for k in ["EmployeeID","Gender","Department","JobLevel","PerformanceLabel","PerformanceScore"]:
        e.pop(k, None)
    X = np.array([e.get(f, 0) for f in feature_names]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred   = model.predict(X_scaled)[0]
    probs  = model.predict_proba(X_scaled)[0]
    labels = {1:"Low", 2:"Medium", 3:"High"}
    return labels[pred], {"Low": float(probs[0]), "Medium": float(probs[1]), "High": float(probs[2])}

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Employee Performance Predictor")
    st.markdown("---")
    st.markdown("**Project Info**")
    st.markdown("""
- 📊 Dataset: 1,000 employees  
- 🔢 Features: 29 (incl. engineered)  
- 🤖 Models: LR, DT, RF, GB, SVM  
- 🎯 Best accuracy: **76.0%**  
- 📁 Built with: Python + Scikit-Learn  
    """)
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.code("pandas · numpy\nscikit-learn\nmatplotlib · seaborn\nstreamlit", language="text")
    st.markdown("---")
    st.caption("Portfolio project · Built for placement prep")

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🧠 Employee Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ML-powered HR analytics · Predict Low / Medium / High performance</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
df = load_dataset()
model, scaler, feature_names, meta = load_model()

model_loaded = model is not None

# ══════════════════════════════════════════════════════════════════════════════
#  TOP METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Employees",  f"{len(df):,}")
c2.metric("Departments",       df["Department"].nunique())
c3.metric("Features",          "29")
c4.metric("Best Accuracy",     f"{meta['test_accuracy']*100:.1f}%" if meta else "76.0%")
c5.metric("F1 Score",          f"{meta['f1_score']*100:.1f}%"      if meta else "75.8%")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔍 EDA Charts",
    "🤖 Model Results",
    "🎯 Predict Employee",
    "📋 HR Report"
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Performance Distribution**")
        counts = df["PerformanceLabel"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = {"High": "#55A868", "Medium": "#4C72B0", "Low": "#C44E52"}
        bars = ax.bar(counts.index, counts.values,
                      color=[colors.get(i, "#888") for i in counts.index],
                      edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Number of Employees", fontsize=11)
        ax.set_title("Performance Category Distribution", fontsize=12, fontweight="bold")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                    str(int(bar.get_height())), ha="center", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Department Breakdown**")
        dept_counts = df["Department"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(dept_counts.index, dept_counts.values, color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Number of Employees", fontsize=11)
        ax.set_title("Employees per Department", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(dept_counts.values):
            ax.text(v + 2, i, str(v), va="center", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Raw Dataset Preview**")
    st.dataframe(df.head(20), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Basic Statistics**")
        st.dataframe(df.describe().round(2), use_container_width=True)
    with col4:
        st.markdown("**Missing Values**")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        }).query("`Missing Count` > 0")
        if len(missing_df):
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values in dataset!")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — EDA CHARTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training Hours by Performance Level**")
        fig, ax = plt.subplots(figsize=(5, 4))
        order   = ["Low", "Medium", "High"]
        palette = {"Low": "#C44E52", "Medium": "#4C72B0", "High": "#55A868"}
        data_clean = df.dropna(subset=["TrainingHours"])
        groups = [data_clean[data_clean["PerformanceLabel"] == lvl]["TrainingHours"].dropna() for lvl in order]
        bp = ax.boxplot(groups, labels=order, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
        for patch, lbl in zip(bp["boxes"], order):
            patch.set_facecolor(palette[lbl])
        ax.set_ylabel("Training Hours / Year")
        ax.set_title("Training Hours vs Performance", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Job Satisfaction Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        for lbl, color in [("High","#55A868"),("Medium","#4C72B0"),("Low","#C44E52")]:
            subset = df[df["PerformanceLabel"] == lbl]["JobSatisfaction"].dropna()
            ax.hist(subset, alpha=0.6, label=lbl, color=color, bins=5, edgecolor="white")
        ax.set_xlabel("Job Satisfaction Score (1–5)")
        ax.set_ylabel("Count")
        ax.set_title("Job Satisfaction by Performance", fontsize=12, fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Salary Distribution by Job Level**")
        fig, ax = plt.subplots(figsize=(5, 4))
        level_order = ["Junior", "Mid", "Senior", "Lead", "Manager"]
        avg_salary  = df.groupby("JobLevel")["Salary"].mean().reindex(level_order)
        ax.bar(avg_salary.index, avg_salary.values, color="#7F77DD", edgecolor="white")
        ax.set_ylabel("Average Salary ($)")
        ax.set_title("Salary by Job Level", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("**Absences vs Review Score**")
        fig, ax = plt.subplots(figsize=(5, 4))
        color_map = {"Low": "#C44E52", "Medium": "#4C72B0", "High": "#55A868"}
        for lbl, grp in df.groupby("PerformanceLabel"):
            ax.scatter(grp["Absences"], grp["LastReviewScore"],
                       c=color_map[lbl], label=lbl, alpha=0.5, s=25)
        ax.set_xlabel("Absences per Year")
        ax.set_ylabel("Last Review Score (1–10)")
        ax.set_title("Absences vs Review Score", fontsize=12, fontweight="bold")
        ax.legend(title="Performance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 3 — full width correlation heatmap
    st.markdown("**Correlation Heatmap**")
    key_cols = ["TrainingHours","ProjectsCompleted","JobSatisfaction",
                "WorkLifeBalance","ManagerRating","Absences",
                "OvertimeHours","PromotionsLast5Yr","LastReviewScore","PerformanceScore"]
    key_cols = [c for c in key_cols if c in df.columns]
    corr = df[key_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Department stacked bar
    st.markdown("**Department × Performance (% stacked)**")
    dept_perf     = df.groupby(["Department","PerformanceLabel"]).size().unstack(fill_value=0)
    dept_perf_pct = dept_perf.div(dept_perf.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    dept_perf_pct.plot(kind="bar", stacked=True, ax=ax,
                       color=["#C44E52","#4C72B0","#55A868"])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Performance Distribution by Department", fontsize=12, fontweight="bold")
    ax.legend(title="Performance", loc="upper right")
    plt.xticks(rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Model Training & Evaluation</div>', unsafe_allow_html=True)

    # Model comparison table (hardcoded from terminal run)
    results_data = {
        "Model":              ["Logistic Regression","SVM","Gradient Boosting","Random Forest","Decision Tree"],
        "Test Accuracy":      [0.760, 0.720, 0.700, 0.695, 0.600],
        "F1 Score (Weighted)":[0.758, 0.704, 0.694, 0.652, 0.590],
        "CV Mean":            [0.729, 0.694, 0.705, 0.696, 0.595],
        "CV Std":             [0.037, 0.036, 0.017, 0.018, 0.056],
    }
    results_df = pd.DataFrame(results_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model",    meta["best_model"]                  if meta else "Logistic Regression")
    col2.metric("Test Accuracy", f"{meta['test_accuracy']*100:.1f}%" if meta else "76.0%")
    col3.metric("F1 Score",      f"{meta['f1_score']*100:.1f}%"      if meta else "75.8%")

    st.markdown("**All Model Results**")
    def color_best(val):
        if isinstance(val, float) and val == results_df["Test Accuracy"].max():
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        return ""
    st.dataframe(
        results_df.style.format({
            "Test Accuracy": "{:.1%}",
            "F1 Score (Weighted)": "{:.1%}",
            "CV Mean": "{:.1%}",
            "CV Std":  "{:.3f}",
        }),
        use_container_width=True
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Accuracy Comparison**")
        fig, ax = plt.subplots(figsize=(5, 4))
        x     = np.arange(len(results_df))
        width = 0.25
        ax.bar(x - width, results_df["Test Accuracy"],       width, label="Test Acc",  color="#4C72B0", edgecolor="white")
        ax.bar(x,         results_df["F1 Score (Weighted)"], width, label="F1 Score",  color="#55A868", edgecolor="white")
        ax.bar(x + width, results_df["CV Mean"],             width, label="CV Mean",   color="#DD8452", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0.5, 0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
        ax.axhline(0.75, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.legend(fontsize=9)
        ax.set_title("Model Performance Comparison", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Confusion Matrix — Best Model**")
        cm = np.array([[18, 11, 2],
                       [6,  87, 15],
                       [3,  11, 47]])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low","Medium","High"],
                    yticklabels=["Low","Medium","High"], ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Feature Importance (Random Forest)**")
    feature_importance_data = {
        "Feature":    ["LastReviewScore","TrainingHours","ProjectsCompleted","TrainingEfficiency",
                       "OverallSatisfaction","AbsenceRate","JobSatisfaction","ManagerRating",
                       "YearsAtCompany","PromotionsLast5Yr"],
        "Importance": [0.134, 0.071, 0.064, 0.050, 0.049, 0.048, 0.043, 0.040, 0.038, 0.034]
    }
    fi_df = pd.DataFrame(feature_importance_data).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Feature Importances", fontsize=12, fontweight="bold")
    for i, v in enumerate(fi_df["Importance"]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — PREDICT EMPLOYEE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Predict a New Employee\'s Performance</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Model not found. Run `python main.py` first to train and save the model.")
    else:
        st.markdown("Fill in the employee details below and click **Predict**.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics & Role**")
            age         = st.number_input("Age",               min_value=18, max_value=65,  value=32)
            gender      = st.selectbox("Gender",              ["Male","Female","Other"])
            department  = st.selectbox("Department",          ["IT","Sales","HR","Finance","Marketing","Operations","R&D"])
            job_level   = st.selectbox("Job Level",           ["Junior","Mid","Senior","Lead","Manager"], index=2)
            years_co    = st.number_input("Years at Company",  min_value=0,  max_value=30,  value=5)
            total_exp   = st.number_input("Total Experience",  min_value=0,  max_value=40,  value=8)

        with col2:
            st.markdown("**Work Metrics**")
            salary      = st.number_input("Salary ($)",         min_value=20000, max_value=250000, value=95000, step=1000)
            training    = st.slider("Training Hours / Year",    0, 100, 75)
            projects    = st.slider("Projects Completed",       1, 30,  18)
            overtime    = st.slider("Overtime Hours / Week",    0, 20,   8)
            absences    = st.slider("Absences / Year",          0, 30,   3)
            promotions  = st.slider("Promotions (last 5 yrs)", 0,  4,   2)

        with col3:
            st.markdown("**Ratings & Scores**")
            job_sat     = st.slider("Job Satisfaction (1–5)",   1,  5, 4)
            wlb         = st.slider("Work-Life Balance (1–5)",  1,  5, 4)
            mgr_rating  = st.slider("Manager Rating (1–5)",     1,  5, 5)
            review      = st.slider("Last Review Score (1–10)", 1.0, 10.0, 8.5, step=0.1)
            distance    = st.number_input("Distance from Office (km)", min_value=1, max_value=100, value=12)

        st.markdown("---")
        predict_btn = st.button("🎯 Predict Performance", type="primary", use_container_width=False)

        if predict_btn:
            emp_data = {
                "Age": age, "Gender": gender, "Department": department,
                "JobLevel": job_level, "YearsAtCompany": years_co,
                "TotalExperience": total_exp, "Salary": salary,
                "TrainingHours": training, "ProjectsCompleted": projects,
                "OvertimeHours": overtime, "JobSatisfaction": job_sat,
                "WorkLifeBalance": wlb, "ManagerRating": mgr_rating,
                "Absences": absences, "PromotionsLast5Yr": promotions,
                "LastReviewScore": review, "DistanceFromOffice": distance
            }

            label, probs = predict_single(emp_data, model, scaler, feature_names)

            badge_color = {"High":"#d4edda","Medium":"#cce5ff","Low":"#f8d7da"}
            text_color  = {"High":"#155724","Medium":"#004085","Low":"#721c24"}
            emoji_map   = {"High":"⭐","Medium":"📊","Low":"⚠️"}

            st.markdown(f"""
            <div style="background:{badge_color[label]};border-radius:12px;padding:1.2rem 1.5rem;margin:1rem 0">
                <div style="font-size:1.6rem;font-weight:700;color:{text_color[label]}">
                    {emoji_map[label]} Predicted Performance: {label}
                </div>
                <div style="color:{text_color[label]};margin-top:0.3rem;font-size:0.95rem">
                    Confidence: {max(probs.values())*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            colA, colB = st.columns(2)

            with colA:
                st.markdown("**Probability Breakdown**")
                prob_colors = {"Low":"#C44E52","Medium":"#4C72B0","High":"#55A868"}
                fig, ax = plt.subplots(figsize=(5, 3))
                labels_list = list(probs.keys())
                vals        = list(probs.values())
                bars = ax.barh(labels_list, vals,
                               color=[prob_colors[l] for l in labels_list], edgecolor="white")
                ax.set_xlim(0, 1)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
                ax.set_title("Prediction Probabilities", fontsize=11, fontweight="bold")
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f"{val*100:.1f}%", va="center", fontsize=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            with colB:
                st.markdown("**HR Recommendation**")
                advice = {
                    "High":   ("Promotion Track", "#d4edda", "#155724",
                               "🚀 This employee is a high performer.\n\n"
                               "• Consider for promotion or leadership role\n"
                               "• Assign to critical / high-visibility projects\n"
                               "• Offer retention bonus or recognition award\n"
                               "• Mentor junior team members"),
                    "Medium": ("Development Plan", "#fff3cd", "#856404",
                               "📈 This employee shows solid potential.\n\n"
                               "• Enroll in targeted skill development program\n"
                               "• Increase 1-on-1 feedback frequency\n"
                               "• Set clear 90-day performance goals\n"
                               "• Assign a senior mentor"),
                    "Low":    ("Performance Review", "#f8d7da", "#721c24",
                               "⚠️ This employee needs immediate attention.\n\n"
                               "• Schedule urgent review with manager\n"
                               "• Create a Performance Improvement Plan (PIP)\n"
                               "• Identify root causes: workload, fit, personal\n"
                               "• Weekly check-ins for next 30 days"),
                }
                title, bg, color, text = advice[label]
                st.markdown(f"""
                <div style="background:{bg};border-radius:10px;padding:1rem 1.2rem">
                    <div style="font-weight:600;color:{color};margin-bottom:0.5rem">{title}</div>
                    <div style="color:{color};font-size:0.9rem;white-space:pre-line">{text}</div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — HR REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">HR Batch Prediction Report</div>', unsafe_allow_html=True)

    pred_path = "outputs/predictions.csv"
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
    else:
        st.warning("Run `python main.py` first to generate predictions.")
        pred_df = None

    if pred_df is not None:
        total   = len(pred_df)
        n_high  = (pred_df["PredictedPerformance"] == "High").sum()
        n_med   = (pred_df["PredictedPerformance"] == "Medium").sum()
        n_low   = (pred_df["PredictedPerformance"] == "Low").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed",   total)
        c2.metric("High Performers",  n_high,  delta="Promotion candidates")
        c3.metric("Medium Performers",n_med)
        c4.metric("At-Risk (Low)",    n_low,   delta="Need PIP", delta_color="inverse")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Predicted Performance Split**")
            counts = pred_df["PredictedPerformance"].value_counts()
            fig, ax = plt.subplots(figsize=(4.5, 4))
            colors = [{"High":"#55A868","Medium":"#4C72B0","Low":"#C44E52"}.get(i,"gray")
                      for i in counts.index]
            ax.pie(counts.values, labels=counts.index, colors=colors,
                   autopct="%1.1f%%", startangle=90,
                   wedgeprops={"edgecolor":"white","linewidth":2})
            ax.set_title("Batch Prediction Results", fontsize=11, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            if "Department" in pred_df.columns:
                st.markdown("**High Performers by Department**")
                high_dept = (pred_df[pred_df["PredictedPerformance"] == "High"]
                             .groupby("Department").size().sort_values())
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ax.barh(high_dept.index, high_dept.values, color="#55A868", edgecolor="white")
                ax.set_xlabel("Count")
                ax.set_title("High Performers by Dept", fontsize=11, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("**Prediction Results Table**")
        display_cols = ["Department","JobLevel","Salary","TrainingHours",
                        "LastReviewScore","PredictedPerformance","PredictionConfidence"]
        display_cols = [c for c in display_cols if c in pred_df.columns]
        show_df = pred_df[display_cols].copy()
        if "PredictionConfidence" in show_df.columns:
            show_df["PredictionConfidence"] = show_df["PredictionConfidence"].apply(lambda x: f"{x*100:.1f}%")

        def highlight_pred(val):
            if val == "High":   return "background-color:#d4edda;color:#155724;font-weight:600"
            if val == "Medium": return "background-color:#cce5ff;color:#004085;font-weight:600"
            if val == "Low":    return "background-color:#f8d7da;color:#721c24;font-weight:600"
            return ""
        st.dataframe(
            show_df.style.applymap(highlight_pred, subset=["PredictedPerformance"]),
            use_container_width=True, height=420
        )

        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv, file_name="hr_predictions.csv", mime="text/csv"
        )
