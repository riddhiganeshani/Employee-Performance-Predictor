"""
predict.py
----------
Prediction module. Loads trained model and makes predictions
on new employee data. Simulates real HR system usage.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os


def load_model_and_scaler():
    """Load saved model, scaler, and feature names."""
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"Model loaded: {metadata['best_model']}")
    print(f"Accuracy: {metadata['test_accuracy']:.4f}")
    
    return model, scaler, feature_names, metadata


def prepare_single_employee(employee_data, feature_names):
    """
    Prepare a single employee record for prediction.
    Applies same encoding and feature engineering as training.
    
    employee_data: dict with employee attributes
    """
    # Feature engineering
    emp = dict(employee_data)
    
    # Derived features
    emp["SalaryPerExperience"] = emp["Salary"] / (emp["TotalExperience"] + 1)
    emp["TrainingEfficiency"] = emp["ProjectsCompleted"] / (emp["TrainingHours"] + 1)
    emp["OverallSatisfaction"] = (
        emp["JobSatisfaction"] + emp["WorkLifeBalance"] + emp["ManagerRating"]
    ) / 3
    emp["AbsenceRate"] = emp["Absences"] / 365 * 100
    emp["CompanyLoyalty"] = emp["YearsAtCompany"] / (emp["TotalExperience"] + 1)
    emp["PromotionVelocity"] = emp["PromotionsLast5Yr"] / (emp["YearsAtCompany"] + 1)
    
    # Encode job level
    job_level_order = {"Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Manager": 5}
    emp["JobLevel_Encoded"] = job_level_order.get(emp.get("JobLevel", "Mid"), 2)
    
    # Encode gender
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    emp["Gender_Encoded"] = gender_map.get(emp.get("Gender", "Male"), 0)
    
    # One-hot encode Department
    departments = ["Sales", "IT", "HR", "Finance", "Marketing", "Operations", "R&D"]
    for dept in departments:
        emp[f"Dept_{dept}"] = 1 if emp.get("Department") == dept else 0
    
    # Remove non-feature keys
    remove_keys = ["EmployeeID", "Gender", "Department", "JobLevel",
                   "PerformanceLabel", "PerformanceScore"]
    for key in remove_keys:
        emp.pop(key, None)
    
    # Build feature vector in correct order
    feature_vector = []
    for feat in feature_names:
        feature_vector.append(emp.get(feat, 0))
    
    return np.array(feature_vector).reshape(1, -1)


def predict_performance(employee_data, model, scaler, feature_names):
    """
    Predict performance for a single employee.
    Returns prediction label and confidence scores.
    """
    label_map = {1: "Low", 2: "Medium", 3: "High"}
    
    # Prepare features
    X = prepare_single_employee(employee_data, feature_names)
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    result = {
        "predicted_performance": label_map[prediction],
        "performance_code": int(prediction),
        "confidence": {
            "Low": round(float(probabilities[0]), 3),
            "Medium": round(float(probabilities[1]), 3),
            "High": round(float(probabilities[2]), 3)
        }
    }
    
    return result


def batch_predict(df_new, model, scaler, feature_names):
    """
    Predict performance for a batch of employees.
    Returns the dataframe with predictions added.
    """
    # Fill missing values before prediction
    df_clean = df_new.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    predictions = []
    confidences = []
    
    for _, row in df_clean.iterrows():
        emp_data = row.to_dict()
        result = predict_performance(emp_data, model, scaler, feature_names)
        predictions.append(result["predicted_performance"])
        confidences.append(max(result["confidence"].values()))
    
    df_new = df_new.copy()
    df_new["PredictedPerformance"] = predictions
    df_new["PredictionConfidence"] = confidences
    
    return df_new


def generate_hr_report(df_predicted):
    """
    Generate an HR insights report from predictions.
    This simulates what HR would receive from the system.
    """
    print("\n" + "=" * 60)
    print("HR PERFORMANCE PREDICTION REPORT")
    print("=" * 60)
    
    total = len(df_predicted)
    
    print(f"\nTotal employees analyzed: {total}")
    
    pred_counts = df_predicted["PredictedPerformance"].value_counts()
    print("\nPredicted Performance Distribution:")
    for label, count in pred_counts.items():
        pct = count / total * 100
        print(f"  {label:8s}: {count:4d} employees ({pct:.1f}%)")
    
    # High performers by department
    if "Department" in df_predicted.columns:
        print("\nHigh Performers by Department:")
        high_perf = df_predicted[df_predicted["PredictedPerformance"] == "High"]
        dept_counts = high_perf.groupby("Department").size().sort_values(ascending=False)
        for dept, count in dept_counts.items():
            print(f"  {dept:15s}: {count} high performers")
    
    # At-risk employees (Low performance)
    low_perf = df_predicted[df_predicted["PredictedPerformance"] == "Low"]
    print(f"\n⚠️  At-Risk Employees (Low Performance): {len(low_perf)}")
    
    if len(low_perf) > 0 and "Department" in df_predicted.columns:
        print("  Distribution:")
        for dept, cnt in low_perf["Department"].value_counts().items():
            print(f"    {dept}: {cnt}")
    
    # Save predictions
    os.makedirs("outputs", exist_ok=True)
    df_predicted.to_csv("outputs/predictions.csv", index=False)
    print("\nPredictions saved to outputs/predictions.csv")
    
    print("\n" + "=" * 60)
    print("HR RECOMMENDATIONS:")
    print("=" * 60)
    print("""
  HIGH Performers:
    → Consider for promotion/leadership roles
    → Offer retention bonuses / recognition
    → Assign to critical projects

  MEDIUM Performers:
    → Provide targeted training programs
    → Set clear performance improvement goals
    → Increase mentoring and feedback frequency

  LOW Performers:
    → Immediate 1-on-1 review with manager
    → Create a Performance Improvement Plan (PIP)
    → Investigate root causes: job fit, workload, personal issues
    """)


def simulate_new_employee():
    """
    Simulate predicting performance for a brand new employee.
    This is the main demo function for the project.
    """
    print("\n" + "=" * 60)
    print("INDIVIDUAL EMPLOYEE PREDICTION DEMO")
    print("=" * 60)
    
    # Sample new employee data
    new_employee = {
        "EmployeeID": "EMP9999",
        "Age": 32,
        "Gender": "Female",
        "Department": "IT",
        "JobLevel": "Senior",
        "YearsAtCompany": 5,
        "TotalExperience": 8,
        "Salary": 95000,
        "TrainingHours": 75,
        "ProjectsCompleted": 18,
        "OvertimeHours": 8,
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
        "ManagerRating": 5,
        "Absences": 3,
        "PromotionsLast5Yr": 2,
        "LastReviewScore": 8.5,
        "DistanceFromOffice": 12,
    }
    
    print("\nEmployee Profile:")
    for key, val in new_employee.items():
        print(f"  {key:25s}: {val}")
    
    return new_employee


if __name__ == "__main__":
    from src.generate_data import generate_employee_data
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load model
    model, scaler, feature_names, metadata = load_model_and_scaler()
    
    # Single employee prediction
    new_emp = simulate_new_employee()
    result = predict_performance(new_emp, model, scaler, feature_names)
    
    print(f"\n{'='*40}")
    print("PREDICTION RESULT:")
    print(f"  Performance: {result['predicted_performance']}")
    print(f"  Confidence Scores:")
    for label, conf in result["confidence"].items():
        print(f"    {label}: {conf:.1%}")
    print(f"{'='*40}")
    
    # Batch prediction on new data
    print("\nRunning batch prediction on 100 new employees...")
    df_new = generate_employee_data(n_employees=100, random_state=99)
    df_with_preds = batch_predict(df_new, model, scaler, feature_names)
    generate_hr_report(df_with_preds)
