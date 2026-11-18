"""
Predicting Student Performance using Supervised Learning
========================================================
This script loads the student performance dataset, trains multiple supervised
learning models for both Regression (final score) and Classification (Pass/Fail),
compares their performance, saves the best models, and creates summary charts.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

FEATURE_COLUMNS = [
    "study_hours",
    "attendance_rate",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "extra_curricular",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from CSV and display a quick overview."""
    df = pd.read_csv(path)
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total records : {len(df)}")
    print(f"Total features: {len(FEATURE_COLUMNS)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("\nPass / Fail Distribution:")
    print(df["result"].value_counts().to_string())
    return df


def train_regression_models(X_train, X_test, y_train, y_test):
    """Train and evaluate supervised regression models to predict final_score."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    }

    results = []
    best_model = None
    best_name = ""
    best_r2 = -np.inf
    best_preds = None

    print("\n" + "=" * 60)
    print("2. REGRESSION MODELS (Predicting Final Score)")
    print("=" * 60)
    print(f"{'Model':<22} {'R2 Score':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
        print(f"{name:<22} {r2:>10.4f} {mae:>10.2f} {rmse:>10.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            best_preds = preds

    print(f"\nBest Regression Model: {best_name} (R2 = {best_r2:.4f})")
    return pd.DataFrame(results), best_name, best_model, best_preds, models["Random Forest"]


def train_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate supervised classification models to predict Pass/Fail."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }

    results = []
    best_model = None
    best_name = ""
    best_acc = -np.inf
    best_preds = None

    print("\n" + "=" * 60)
    print("3. CLASSIFICATION MODELS (Predicting Pass / Fail)")
    print("=" * 60)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 45)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label="Pass")

        results.append({"Model": name, "Accuracy": acc, "F1": f1})
        print(f"{name:<22} {acc * 100:>9.2f}% {f1:>10.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            best_preds = preds

    print(f"\nBest Classification Model: {best_name} (Accuracy = {best_acc * 100:.2f}%)")
    print("\nDetailed Classification Report (Best Model):")
    print(classification_report(y_test, best_preds))
    return pd.DataFrame(results), best_name, best_model


def save_plots(df, reg_results, clf_results, y_test_reg, best_reg_preds, best_reg_name, rf_regressor):
    """Generate and save easy-to-read evaluation charts in outputs/."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_cols = FEATURE_COLUMNS + ["final_score"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Feature Correlation with Final Exam Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 2. Model Comparison (Regression R2 & Classification Accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.barplot(data=reg_results, x="Model", y="R2", hue="Model", palette="Blues_d", legend=False, ax=axes[0])
    axes[0].set_title("Regression Models (R2 Score)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R2 Score (Higher is Better)")
    axes[0].set_xlabel("")
    for idx, row in reg_results.iterrows():
        axes[0].text(idx, row["R2"] + 0.02, f"{row['R2']:.3f}", ha="center", fontweight="bold")

    clf_plot = clf_results.copy()
    clf_plot["AccuracyPct"] = clf_plot["Accuracy"] * 100
    sns.barplot(data=clf_plot, x="Model", y="AccuracyPct", hue="Model", palette="Greens_d", legend=False, ax=axes[1])
    axes[1].set_title("Classification Models (Accuracy %)", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("")
    for idx, row in clf_plot.iterrows():
        axes[1].text(idx, row["AccuracyPct"] + 1.8, f"{row['AccuracyPct']:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # 3. Actual vs Predicted Final Score
    plt.figure(figsize=(7, 5.5))
    plt.scatter(y_test_reg, best_reg_preds, alpha=0.7, color="#2563eb", edgecolors="white", s=55)
    min_val = min(y_test_reg.min(), best_reg_preds.min()) - 3
    max_val = max(y_test_reg.max(), best_reg_preds.max()) + 3
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    plt.title(f"Actual vs Predicted Final Score ({best_reg_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 4. Feature Importance (from Random Forest Regressor)
    importances = pd.Series(rf_regressor.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(8, 4.8))
    importances.plot(kind="barh", color="#0ea5e9")
    plt.title("Feature Importance for Predicting Student Score", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    print(f"\nCharts saved in: {OUTPUTS_DIR}")


def main():
    # Step 1: Load data
    df = load_data(DATA_PATH)

    # Step 2: Separate features and targets
    X = df[FEATURE_COLUMNS]
    y_score = df["final_score"]
    y_result = df["result"]

    # Step 3: Train/Test split (80% train, 20% test)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_result, test_size=0.2, random_state=42, stratify=y_result
    )

    # Step 4: Train and compare models
    reg_results, best_reg_name, best_reg_model, best_reg_preds, rf_reg = train_regression_models(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )
    clf_results, best_clf_name, best_clf_model = train_classification_models(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf
    )

    # Step 5: Save best models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_reg_model, MODELS_DIR / "score_model.pkl")
    joblib.dump(best_clf_model, MODELS_DIR / "pass_fail_model.pkl")
    print("=" * 60)
    print("4. SAVED MODELS")
    print("=" * 60)
    print(f"Regression model ({best_reg_name})      -> models/score_model.pkl")
    print(f"Classification model ({best_clf_name}) -> models/pass_fail_model.pkl")

    # Step 6: Save visualizations
    save_plots(df, reg_results, clf_results, y_test_reg, best_reg_preds, best_reg_name, rf_reg)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
