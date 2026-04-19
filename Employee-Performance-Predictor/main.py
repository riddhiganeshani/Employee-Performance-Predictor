"""
main.py
-------
Main entry point for the Employee Performance Predictor project.
Run this file to execute the complete pipeline end-to-end.

Usage:
    python main.py              # Run full pipeline
    python main.py --eda        # Run EDA only
    python main.py --train      # Train model only
    python main.py --predict    # Run prediction only
"""

import argparse
import os
import sys

# Make sure src imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_full_pipeline():
    """Run the entire pipeline from data generation to prediction."""
    print("\n" + "🚀 " * 20)
    print("EMPLOYEE PERFORMANCE PREDICTOR - FULL PIPELINE")
    print("🚀 " * 20 + "\n")
    
    # ─── STEP 1: Generate Data ────────────────────────────────────────────────
    print("\n[STEP 1/5] Generating Synthetic Employee Dataset...")
    from src.generate_data import generate_employee_data
    df = generate_employee_data(n_employees=1000)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/employee_data.csv", index=False)
    print(f"  ✅ Dataset created: {df.shape[0]} employees, {df.shape[1]} features")
    
    # ─── STEP 2: EDA ─────────────────────────────────────────────────────────
    print("\n[STEP 2/5] Running Exploratory Data Analysis...")
    from src.eda import run_eda
    run_eda("data/employee_data.csv")
    print("  ✅ EDA complete - charts saved to images/")
    
    # ─── STEP 3: Train Model ──────────────────────────────────────────────────
    print("\n[STEP 3/5] Training Machine Learning Models...")
    from src.train_model import run_training
    best_model, feature_names, scaler = run_training()
    print("  ✅ Model training complete - best model saved to models/")
    
    # ─── STEP 4: Predictions ──────────────────────────────────────────────────
    print("\n[STEP 4/5] Running Predictions...")
    from src.predict import (
        load_model_and_scaler, simulate_new_employee,
        predict_performance, batch_predict, generate_hr_report
    )
    
    model, scaler, feature_names, metadata = load_model_and_scaler()
    
    # Single prediction demo
    new_emp = simulate_new_employee()
    result = predict_performance(new_emp, model, scaler, feature_names)
    
    print(f"\n  📊 Sample Prediction Result:")
    print(f"     Employee: {new_emp['EmployeeID']} | Dept: {new_emp['Department']} | Level: {new_emp['JobLevel']}")
    print(f"     Predicted Performance: ⭐ {result['predicted_performance']}")
    print(f"     Confidence: {max(result['confidence'].values()):.1%}")
    
    # Batch prediction
    df_new = generate_employee_data(n_employees=100, random_state=99)
    df_predicted = batch_predict(df_new, model, scaler, feature_names)
    generate_hr_report(df_predicted)
    
    # ─── STEP 5: Summary ─────────────────────────────────────────────────────
    print("\n[STEP 5/5] Pipeline Summary")
    print_final_summary(metadata)


def print_final_summary(metadata=None):
    """Print final project summary."""
    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETE")
    print("=" * 60)
    
    if metadata:
        print(f"\n📈 Best Model:    {metadata['best_model']}")
        print(f"🎯 Test Accuracy: {metadata['test_accuracy']:.2%}")
        print(f"📊 F1 Score:      {metadata['f1_score']:.2%}")
    
    print("""
📁 Output Files Generated:
   data/
     ├── employee_data.csv           (raw dataset)
     └── employee_data_preprocessed.csv (cleaned)
   
   models/
     ├── best_model.pkl              (trained model)
     ├── scaler.pkl                  (feature scaler)
     ├── feature_names.pkl           (feature list)
     └── model_metadata.json         (accuracy info)
   
   outputs/
     ├── predictions.csv             (HR predictions)
     ├── model_comparison.csv        (all models eval)
     └── summary_statistics.csv      (dataset stats)
   
   images/
     ├── 01_performance_distribution.png
     ├── 02_department_performance.png
     ├── 03_age_salary_distribution.png
     ├── 04_training_vs_performance.png
     ├── 05_correlation_heatmap.png
     ├── 06_salary_by_level.png
     ├── 07_absences_vs_review.png
     ├── 08_model_comparison.png
     ├── 09_confusion_matrix.png
     └── 10_feature_importance.png
""")
    print("=" * 60)
    print("✅ Ready for GitHub upload and interview demo!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Employee Performance Predictor"
    )
    parser.add_argument("--eda", action="store_true", help="Run EDA only")
    parser.add_argument("--train", action="store_true", help="Train model only")
    parser.add_argument("--predict", action="store_true", help="Run prediction only")
    
    args = parser.parse_args()
    
    if args.eda:
        from src.eda import run_eda
        run_eda("data/employee_data.csv")
    elif args.train:
        from src.train_model import run_training
        run_training()
    elif args.predict:
        from src.predict import (
            load_model_and_scaler, simulate_new_employee,
            predict_performance, batch_predict, generate_hr_report
        )
        from src.generate_data import generate_employee_data
        model, scaler, feature_names, metadata = load_model_and_scaler()
        new_emp = simulate_new_employee()
        result = predict_performance(new_emp, model, scaler, feature_names)
        print(f"\nPrediction: {result['predicted_performance']}")
        print(f"Confidence: {result['confidence']}")
        
        df_new = generate_employee_data(100, random_state=99)
        df_pred = batch_predict(df_new, model, scaler, feature_names)
        generate_hr_report(df_pred)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
