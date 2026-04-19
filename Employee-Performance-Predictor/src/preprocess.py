"""
preprocess.py
-------------
Handles all data cleaning and feature engineering for the employee dataset.
Includes missing value treatment, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os


def load_data(filepath="data/employee_data.csv"):
    """Load raw employee data."""
    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_missing_values(df):
    """Report missing values in dataset."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct.round(2)
    })
    report = report[report["Missing Count"] > 0]
    print("\nMissing Values Report:")
    print(report)
    return report


def handle_missing_values(df):
    """
    Fill missing values using appropriate strategies:
    - Numeric: fill with median (robust to outliers)
    - Categorical: fill with mode
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled '{col}' missing values with median: {median_val}")
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  Filled '{col}' missing values with mode: {mode_val}")
    
    return df


def feature_engineering(df):
    """
    Create new meaningful features from existing ones.
    This is the creative part of data science!
    """
    df = df.copy()
    
    # Salary per year of experience
    df["SalaryPerExperience"] = (df["Salary"] / (df["TotalExperience"] + 1)).round(2)
    
    # Training efficiency: projects per training hour
    df["TrainingEfficiency"] = (df["ProjectsCompleted"] / (df["TrainingHours"] + 1)).round(2)
    
    # Overall satisfaction score (average of 3 satisfaction metrics)
    df["OverallSatisfaction"] = (
        df["JobSatisfaction"] + df["WorkLifeBalance"] + df["ManagerRating"]
    ) / 3
    df["OverallSatisfaction"] = df["OverallSatisfaction"].round(2)
    
    # Absence rate (normalized)
    df["AbsenceRate"] = (df["Absences"] / 365 * 100).round(2)
    
    # Experience ratio (company experience vs total)
    df["CompanyLoyalty"] = (df["YearsAtCompany"] / (df["TotalExperience"] + 1)).round(2)
    
    # Promotion velocity (promotions per year at company)
    df["PromotionVelocity"] = (
        df["PromotionsLast5Yr"] / (df["YearsAtCompany"] + 1)
    ).round(3)
    
    print("Feature engineering complete. New features added:")
    new_features = [
        "SalaryPerExperience", "TrainingEfficiency", "OverallSatisfaction",
        "AbsenceRate", "CompanyLoyalty", "PromotionVelocity"
    ]
    for f in new_features:
        print(f"  + {f}")
    
    return df


def encode_categorical(df):
    """
    Encode categorical variables for ML model consumption.
    - Label encoding for ordinal variables
    - One-hot encoding for nominal variables
    """
    df = df.copy()
    
    # Label encode ordinal variable
    job_level_order = {"Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Manager": 5}
    df["JobLevel_Encoded"] = df["JobLevel"].map(job_level_order)
    
    # Gender encoding
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    df["Gender_Encoded"] = df["Gender"].map(gender_map)
    
    # One-hot encode Department
    dept_dummies = pd.get_dummies(df["Department"], prefix="Dept")
    df = pd.concat([df, dept_dummies], axis=1)
    
    print(f"Encoding complete. Departments one-hot encoded: {df['Department'].nunique()} categories")
    return df


def prepare_features(df):
    """
    Select final feature set for model training.
    Returns X (features) and y (target).
    """
    # Drop non-feature columns
    drop_cols = [
        "EmployeeID", "Gender", "Department", "JobLevel",
        "PerformanceLabel"  # string version of target
    ]
    
    # Target variable
    y = df["PerformanceScore"].astype(int)
    
    # Feature matrix
    X = df.drop(drop_cols + ["PerformanceScore"], axis=1)
    
    print(f"\nFinal features: {X.shape[1]} columns")
    print(f"Target classes: {sorted(y.unique())} (1=Low, 2=Medium, 3=High)")
    
    return X, y


def scale_features(X_train, X_test):
    """
    Standardize features (mean=0, std=1).
    Fit only on training data to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for production use
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print("Feature scaling complete. Scaler saved to models/scaler.pkl")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath="data/employee_data.csv", test_size=0.2):
    """
    Full preprocessing pipeline - runs all steps in sequence.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE STARTED")
    print("=" * 50)
    
    # Step 1: Load
    df = load_data(filepath)
    
    # Step 2: Check missing values
    check_missing_values(df)
    
    # Step 3: Handle missing values
    print("\nHandling missing values...")
    df = handle_missing_values(df)
    
    # Step 4: Feature engineering
    print("\nRunning feature engineering...")
    df = feature_engineering(df)
    
    # Step 5: Encode categoricals
    print("\nEncoding categorical variables...")
    df = encode_categorical(df)
    
    # Step 6: Prepare X and y
    print("\nPreparing feature matrix...")
    X, y = prepare_features(df)
    feature_names = X.columns.tolist()
    
    # Step 7: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    
    # Step 8: Scale
    print("\nScaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save preprocessed data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/employee_data_preprocessed.csv", index=False)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline()
    print(f"\nReady for model training!")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
