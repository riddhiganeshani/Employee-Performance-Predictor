"""
generate_data.py
----------------
Generates a synthetic HR Employee dataset simulating real company data.
No real data needed - we create realistic, statistically sound employee records.
"""

import pandas as pd
import numpy as np
import os

def generate_employee_data(n_employees=1000, random_state=42):
    """
    Generate synthetic HR employee data with realistic patterns.
    
    Parameters:
        n_employees (int): Number of employee records to generate
        random_state (int): Seed for reproducibility
    
    Returns:
        pd.DataFrame: Complete employee dataset
    """
    np.random.seed(random_state)
    
    # --- Basic Demographics ---
    employee_ids = [f"EMP{str(i).zfill(4)}" for i in range(1, n_employees + 1)]
    
    ages = np.random.randint(22, 60, n_employees)
    
    genders = np.random.choice(["Male", "Female", "Other"], n_employees, p=[0.52, 0.46, 0.02])
    
    departments = np.random.choice(
        ["Sales", "IT", "HR", "Finance", "Marketing", "Operations", "R&D"],
        n_employees, p=[0.20, 0.25, 0.10, 0.15, 0.10, 0.12, 0.08]
    )
    
    job_levels = np.random.choice(
        ["Junior", "Mid", "Senior", "Lead", "Manager"],
        n_employees, p=[0.30, 0.30, 0.20, 0.12, 0.08]
    )
    
    # --- Experience & Tenure ---
    years_at_company = np.random.randint(0, 20, n_employees)
    total_experience = years_at_company + np.random.randint(0, 10, n_employees)
    total_experience = np.clip(total_experience, 0, 35)
    
    # --- Compensation ---
    base_salary_map = {
        "Junior": (30000, 50000),
        "Mid": (50000, 80000),
        "Senior": (80000, 120000),
        "Lead": (100000, 150000),
        "Manager": (120000, 200000)
    }
    
    salaries = []
    for level in job_levels:
        low, high = base_salary_map[level]
        salaries.append(np.random.randint(low, high))
    salaries = np.array(salaries)
    
    # --- Work Metrics ---
    training_hours = np.random.randint(0, 100, n_employees)
    projects_completed = np.random.randint(1, 30, n_employees)
    overtime_hours = np.random.randint(0, 20, n_employees)
    
    # Satisfaction scores (1-5 scale)
    job_satisfaction = np.random.randint(1, 6, n_employees)
    work_life_balance = np.random.randint(1, 6, n_employees)
    manager_rating = np.random.randint(1, 6, n_employees)
    
    # Absences per year
    absences = np.random.randint(0, 30, n_employees)
    
    # Promotions in last 5 years
    promotions_last_5yr = np.random.randint(0, 4, n_employees)
    
    # Last performance review score (1-10)
    last_review_score = np.random.uniform(3, 10, n_employees).round(1)
    
    # Distance from office (km)
    distance_from_office = np.random.randint(1, 100, n_employees)
    
    # --- Target Variable: Performance Score (1=Low, 2=Medium, 3=High) ---
    # Create realistic performance based on multiple factors
    performance_score = np.zeros(n_employees)
    
    for i in range(n_employees):
        score = 0
        
        # Training hours positive impact
        if training_hours[i] > 60: score += 2
        elif training_hours[i] > 30: score += 1
        
        # Projects completed positive impact
        if projects_completed[i] > 20: score += 2
        elif projects_completed[i] > 10: score += 1
        
        # Satisfaction impacts performance
        if job_satisfaction[i] >= 4: score += 1
        elif job_satisfaction[i] <= 2: score -= 1
        
        # Manager rating impact
        if manager_rating[i] >= 4: score += 1
        elif manager_rating[i] <= 2: score -= 1
        
        # Absences hurt performance
        if absences[i] > 20: score -= 2
        elif absences[i] > 10: score -= 1
        
        # Experience helps
        if total_experience[i] > 10: score += 1
        
        # Overtime moderate impact
        if overtime_hours[i] > 10: score += 1
        
        # Promotions indicate past performance
        if promotions_last_5yr[i] >= 2: score += 1
        
        # Last review score
        if last_review_score[i] >= 8: score += 2
        elif last_review_score[i] <= 4: score -= 2
        
        # Add noise
        score += np.random.randint(-1, 2)
        
        # Map to 3-class performance
        if score >= 5:
            performance_score[i] = 3  # High
        elif score >= 1:
            performance_score[i] = 2  # Medium
        else:
            performance_score[i] = 1  # Low
    
    performance_label = {1: "Low", 2: "Medium", 3: "High"}
    performance_labels = [performance_label[int(p)] for p in performance_score]
    
    # --- Build DataFrame ---
    df = pd.DataFrame({
        "EmployeeID": employee_ids,
        "Age": ages,
        "Gender": genders,
        "Department": departments,
        "JobLevel": job_levels,
        "YearsAtCompany": years_at_company,
        "TotalExperience": total_experience,
        "Salary": salaries,
        "TrainingHours": training_hours,
        "ProjectsCompleted": projects_completed,
        "OvertimeHours": overtime_hours,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "ManagerRating": manager_rating,
        "Absences": absences,
        "PromotionsLast5Yr": promotions_last_5yr,
        "LastReviewScore": last_review_score,
        "DistanceFromOffice": distance_from_office,
        "PerformanceScore": performance_score.astype(int),
        "PerformanceLabel": performance_labels
    })
    
    # Introduce ~5% missing values to simulate real data
    for col in ["TrainingHours", "JobSatisfaction", "ManagerRating", "Absences"]:
        mask = np.random.random(n_employees) < 0.05
        df.loc[mask, col] = np.nan
    
    return df


if __name__ == "__main__":
    print("Generating synthetic employee dataset...")
    df = generate_employee_data(n_employees=1000)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/employee_data.csv", index=False)
    
    print(f"Dataset created: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Performance Distribution:\n{df['PerformanceLabel'].value_counts()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSaved to data/employee_data.csv")
