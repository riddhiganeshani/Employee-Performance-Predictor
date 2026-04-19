# 🧠 Employee Performance Predictor using Data Analytics

> **Predicting employee performance with Machine Learning — helping HR teams make data-driven decisions**

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 Project Overview

This project builds an **end-to-end Machine Learning system** that predicts whether an employee will perform at a **Low**, **Medium**, or **High** level, based on HR data such as training hours, satisfaction scores, work history, and more.

Built using **synthetic HR data** (simulating a real company) — ideal for students building portfolio projects without access to private company data.

---

## 🎯 Problem Statement

Companies lose millions annually from:
- Undetected low performers staying in roles too long
- Missing signs of burnout before talented employees leave
- Inefficient promotion decisions
- Poor training investment allocation

**This system solves that** — giving HR teams an AI-powered early warning system.

---

## 💼 Business Value

| Stakeholder | How They Benefit |
|------------|-----------------|
| HR Manager | Identify at-risk employees before performance drops |
| Department Manager | Get data-backed insights on team performance |
| CEO / Leadership | Reduce turnover costs, optimize workforce planning |
| L&D Team | Focus training budget on employees who need it most |
| Recruitment | Benchmark performance profiles for hiring |

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Visualization | Matplotlib, Seaborn |
| Models Used | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM |
| Environment | Jupyter Notebook / VSCode |
| Version Control | Git + GitHub |

---

## 🏗️ Project Architecture

```
Raw Employee Data (Synthetic)
           │
           ▼
   [Data Generation]
   1000 employee records
   20 features per employee
           │
           ▼
   [Data Preprocessing]
   → Missing value treatment
   → Feature engineering (6 new features)
   → Label encoding + One-hot encoding
   → Train/test split (80/20)
   → StandardScaler normalization
           │
           ▼
   [Model Training]
   → 5 ML algorithms trained
   → Cross-validation (5-fold)
   → Best model selected
           │
           ▼
   [Prediction Engine]
   → Single employee prediction
   → Batch prediction (100s of employees)
   → Confidence scores per prediction
           │
           ▼
   [HR Insights Report]
   → Performance distribution
   → High performer identification
   → At-risk employee flagging
   → Department-level breakdown
```

---

## 📁 Folder Structure

```
Employee-Performance-Predictor/
│
├── data/
│   ├── employee_data.csv                   ← Raw synthetic dataset
│   └── employee_data_preprocessed.csv      ← Cleaned & engineered data
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py                    ← Creates synthetic HR data
│   ├── preprocess.py                       ← Cleaning + feature engineering
│   ├── eda.py                              ← All visualizations & stats
│   ├── train_model.py                      ← Model training & evaluation
│   └── predict.py                          ← Prediction engine + HR report
│
├── models/
│   ├── best_model.pkl                      ← Trained ML model
│   ├── scaler.pkl                          ← Feature scaler
│   ├── feature_names.pkl                   ← Input feature list
│   └── model_metadata.json                 ← Accuracy & metrics
│
├── outputs/
│   ├── predictions.csv                     ← Employee predictions
│   ├── model_comparison.csv                ← All models evaluation
│   └── summary_statistics.csv              ← Dataset statistics
│
├── images/
│   ├── 01_performance_distribution.png
│   ├── 02_department_performance.png
│   ├── 03_age_salary_distribution.png
│   ├── 04_training_vs_performance.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_salary_by_level.png
│   ├── 07_absences_vs_review.png
│   ├── 08_model_comparison.png
│   ├── 09_confusion_matrix.png
│   └── 10_feature_importance.png
│
├── notebooks/
│   └── analysis.ipynb                      ← (optional) Jupyter notebook
│
├── main.py                                 ← 🚀 Run this file
├── requirements.txt                        ← Python dependencies
└── README.md                               ← This file
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run Complete Pipeline (Recommended)
```bash
python main.py
```
This runs all 5 steps: data generation → EDA → training → evaluation → prediction.

### Run Individual Steps
```bash
python main.py --eda        # EDA charts only
python main.py --train      # Model training only
python main.py --predict    # Predictions only
```

---

## 📊 Results

| Model | Test Accuracy | F1 Score | CV Mean |
|-------|--------------|----------|---------|
| Random Forest | ~85% | ~0.84 | ~0.84 |
| Gradient Boosting | ~84% | ~0.83 | ~0.83 |
| Logistic Regression | ~78% | ~0.77 | ~0.77 |
| Decision Tree | ~76% | ~0.75 | ~0.74 |
| SVM | ~80% | ~0.79 | ~0.80 |

> **Best Model: Random Forest** (typically wins on this type of structured HR data)

---

## 🖼️ Sample Outputs

After running `python main.py`, you'll get:

- **10 visualization charts** in `images/`
- **Prediction CSV** in `outputs/predictions.csv`
- **HR insights report** printed to console
- **Trained model** saved in `models/`

---

## 🔮 Features of the Dataset

| Feature | Description |
|---------|-------------|
| Age | Employee age (22-60) |
| Department | IT, Sales, HR, Finance, Marketing, Operations, R&D |
| JobLevel | Junior → Manager (5 levels) |
| TrainingHours | Annual training hours |
| ProjectsCompleted | Projects completed per year |
| JobSatisfaction | Self-rated (1-5) |
| ManagerRating | Manager's rating (1-5) |
| Absences | Absences per year |
| LastReviewScore | Performance review score (1-10) |
| PerformanceScore | **TARGET**: 1=Low, 2=Medium, 3=High |

---

## 🔮 Future Improvements

- [ ] Employee Attrition Prediction module
- [ ] Interactive Streamlit dashboard
- [ ] Deep Learning model (Neural Network)
- [ ] Real-time prediction API (FastAPI/Flask)
- [ ] Integration with real HRMS systems
- [ ] SHAP values for explainability

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License.
