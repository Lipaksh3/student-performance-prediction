"""
Student Performance Prediction Script
=====================================
Loads the trained supervised learning models and predicts a student's
expected final exam score and Pass/Fail status.

Usage:
    python src/predict.py          # Interactive mode (enter student details)
    python src/predict.py --demo   # Run predictions on 3 sample students
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCORE_MODEL_PATH = BASE_DIR / "models" / "score_model.pkl"
PASS_FAIL_MODEL_PATH = BASE_DIR / "models" / "pass_fail_model.pkl"

FEATURE_COLUMNS = [
    "study_hours",
    "attendance_rate",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "extra_curricular",
]


def load_models():
    """Load saved regression and classification models."""
    if not SCORE_MODEL_PATH.exists() or not PASS_FAIL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found. Please run 'python src/train_model.py' first."
        )
    score_model = joblib.load(SCORE_MODEL_PATH)
    pass_fail_model = joblib.load(PASS_FAIL_MODEL_PATH)
    return score_model, pass_fail_model


def predict_student(score_model, pass_fail_model, student_data: dict) -> tuple[float, str, float]:
    """Return predicted final score, Pass/Fail label, and Pass probability."""
    input_df = pd.DataFrame([student_data])[FEATURE_COLUMNS]
    predicted_score = float(score_model.predict(input_df)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))

    predicted_result = str(pass_fail_model.predict(input_df)[0])
    pass_index = list(pass_fail_model.classes_).index("Pass")
    pass_prob = float(pass_fail_model.predict_proba(input_df)[0][pass_index]) * 100.0
    return predicted_score, predicted_result, pass_prob


def run_demo(score_model, pass_fail_model):
    """Run predictions on three example student profiles."""
    samples = [
        {
            "name": "High Performer",
            "study_hours": 6.0,
            "attendance_rate": 92.0,
            "previous_score": 84.0,
            "assignments_completed": 9,
            "sleep_hours": 7.5,
            "extra_curricular": 1,
        },
        {
            "name": "Average Student",
            "study_hours": 3.0,
            "attendance_rate": 74.0,
            "previous_score": 58.0,
            "assignments_completed": 6,
            "sleep_hours": 7.0,
            "extra_curricular": 1,
        },
        {
            "name": "At-Risk Student",
            "study_hours": 1.0,
            "attendance_rate": 50.0,
            "previous_score": 38.0,
            "assignments_completed": 2,
            "sleep_hours": 5.0,
            "extra_curricular": 0,
        },
    ]

    print("=" * 60)
    print("SAMPLE STUDENT PREDICTIONS")
    print("=" * 60)
    for sample in samples:
        name = sample["name"]
        score, result, prob = predict_student(score_model, pass_fail_model, sample)
        print(f"\nProfile          : {name}")
        print(
            f"Inputs           : Study={sample['study_hours']}h/day, "
            f"Attendance={sample['attendance_rate']}%, "
            f"PrevScore={sample['previous_score']}, "
            f"Assignments={sample['assignments_completed']}/10"
        )
        print(f"Predicted Score  : {score:.1f} / 100")
        print(f"Predicted Result : {result} (Pass Confidence: {prob:.1f}%)")


def get_float(prompt: str, min_val: float, max_val: float) -> float:
    """Safely read a float within a valid range."""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def run_interactive(score_model, pass_fail_model):
    """Prompt the user for student details and print the prediction."""
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTOR")
    print("=" * 60)

    student_data = {
        "study_hours": get_float("Daily study hours (0.5 - 12): ", 0.5, 12.0),
        "attendance_rate": get_float("Attendance percentage (0 - 100): ", 0.0, 100.0),
        "previous_score": get_float("Previous exam score (0 - 100): ", 0.0, 100.0),
        "assignments_completed": int(get_float("Assignments completed (0 - 10): ", 0, 10)),
        "sleep_hours": get_float("Daily sleep hours (3 - 12): ", 3.0, 12.0),
        "extra_curricular": int(get_float("Extra-curricular activities (0 = No, 1 = Yes): ", 0, 1)),
    }

    score, result, prob = predict_student(score_model, pass_fail_model, student_data)
    print("\n" + "-" * 40)
    print(f"Predicted Final Score : {score:.1f} / 100")
    print(f"Predicted Result      : {result}")
    print(f"Pass Probability      : {prob:.1f}%")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Predict student performance.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run sample predictions without interactive input.",
    )
    args = parser.parse_args()

    score_model, pass_fail_model = load_models()
    if args.demo:
        run_demo(score_model, pass_fail_model)
    else:
        run_interactive(score_model, pass_fail_model)


if __name__ == "__main__":
    main()
