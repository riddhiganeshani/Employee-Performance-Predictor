"""
train_model.py
--------------
Model training module. Trains multiple classifiers, evaluates them,
and saves the best model for production use.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.model_selection import cross_val_score

from src.preprocess import preprocess_pipeline


def train_all_models(X_train, y_train):
    """
    Train multiple ML models and return them in a dictionary.
    This is called a 'model comparison' approach used in industry.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=42
        ),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42, probability=True)
    }
    
    print("Training models...")
    trained = {}
    for name, model in models.items():
        print(f"  Training: {name}...", end=" ")
        model.fit(X_train, y_train)
        trained[name] = model
        print("Done")
    
    return trained


def evaluate_models(trained_models, X_train, X_test, y_train, y_test):
    """
    Evaluate all models and collect metrics.
    Returns a results DataFrame and the best model name.
    """
    results = []
    
    for name, model in trained_models.items():
        # Test set predictions
        y_pred = model.predict(X_test)
        
        # Cross-validation on training set (5-fold)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            "Model": name,
            "Test Accuracy": round(acc, 4),
            "F1 Score (Weighted)": round(f1, 4),
            "CV Mean Accuracy": round(cv_mean, 4),
            "CV Std": round(cv_std, 4)
        })
        
        print(f"\n{name}:")
        print(f"  Test Accuracy: {acc:.4f} | F1: {f1:.4f} | CV: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    
    # Best model = highest test accuracy
    best_model_name = results_df.iloc[0]["Model"]
    
    return results_df, best_model_name


def plot_model_comparison(results_df):
    """Bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    bars1 = ax.bar(x - width, results_df["Test Accuracy"], width,
                   label="Test Accuracy", color="#4C72B0")
    bars2 = ax.bar(x, results_df["F1 Score (Weighted)"], width,
                   label="F1 Score", color="#55A868")
    bars3 = ax.bar(x + width, results_df["CV Mean Accuracy"], width,
                   label="CV Mean", color="#DD8452")
    
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="80% threshold")
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig("images/08_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/08_model_comparison.png")


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Confusion matrix heatmap for best model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"],
                ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    
    plt.tight_layout()
    plt.savefig("images/09_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/09_confusion_matrix.png")


def plot_feature_importance(model, feature_names, model_name):
    """Feature importance chart for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"  (Skipping feature importance - {model_name} doesn't support it)")
        return
    
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#4C72B0")
    ax.set_title(f"Top 15 Feature Importances - {model_name}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    
    plt.tight_layout()
    plt.savefig("images/10_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("Saved: images/10_feature_importance.png")


def save_best_model(model, model_name, results_df):
    """Save the best model and its metadata."""
    os.makedirs("models", exist_ok=True)
    
    # Save model
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save results
    results_df.to_csv("outputs/model_comparison.csv", index=False)
    
    # Save metadata
    best_row = results_df[results_df["Model"] == model_name].iloc[0]
    metadata = {
        "best_model": model_name,
        "test_accuracy": float(best_row["Test Accuracy"]),
        "f1_score": float(best_row["F1 Score (Weighted)"]),
        "cv_mean": float(best_row["CV Mean Accuracy"]),
        "classes": [1, 2, 3],
        "class_labels": {"1": "Low", "2": "Medium", "3": "High"}
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nBest model saved: {model_name}")
    print(f"  Accuracy: {best_row['Test Accuracy']:.4f}")
    print(f"  F1 Score: {best_row['F1 Score (Weighted)']:.4f}")


def print_classification_report(model, X_test, y_test):
    """Print detailed classification report."""
    y_pred = model.predict(X_test)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Low", "Medium", "High"]))


def run_training():
    """Full training pipeline."""
    print("=" * 50)
    print("MODEL TRAINING PIPELINE STARTED")
    print("=" * 50)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline()
    
    print("\n" + "=" * 50)
    print("TRAINING MODELS")
    print("=" * 50)
    
    # Train all models
    trained_models = train_all_models(X_train, y_train)
    
    print("\n" + "=" * 50)
    print("EVALUATING MODELS")
    print("=" * 50)
    
    # Evaluate
    results_df, best_model_name = evaluate_models(
        trained_models, X_train, X_test, y_train, y_test
    )
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print("\nAll Results:")
    print(results_df.to_string(index=False))
    
    best_model = trained_models[best_model_name]
    
    # Detailed report
    print_classification_report(best_model, X_test, y_test)
    
    # Visualizations
    print("\nGenerating evaluation charts...")
    plot_model_comparison(results_df)
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    plot_feature_importance(best_model, feature_names, best_model_name)
    
    # Save
    save_best_model(best_model, best_model_name, results_df)
    
    # Save feature names
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    return best_model, feature_names, scaler


if __name__ == "__main__":
    run_training()
