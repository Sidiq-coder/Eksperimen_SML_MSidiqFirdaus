import os
import time
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TRAIN_PATH = os.path.join("titanic_preprocessing", "train.csv")
TEST_PATH = os.path.join("titanic_preprocessing", "test.csv")
TARGET_COLUMN = "Survived"
ARTIFACT_DIR = "artifacts"


def load_data():
    print("Current working directory:", os.getcwd())
    print("Train path:", TRAIN_PATH)
    print("Test path:", TEST_PATH)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"File train.csv tidak ditemukan: {TRAIN_PATH}")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"File test.csv tidak ditemukan: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Kolom target {TARGET_COLUMN} tidak ada di train.csv")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Kolom target {TARGET_COLUMN} tidak ada di test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    mlflow.set_experiment("Titanic_Workflow_CI")

    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    start_time = time.time()

    with mlflow.start_run(run_name="workflow_ci_titanic_random_forest"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        training_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("dataset", "Titanic preprocessed dataset")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("training_time_seconds", training_time)

        model_path = os.path.join(ARTIFACT_DIR, "titanic_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": accuracy,
                    "precision_weighted": precision,
                    "recall_weighted": recall,
                    "f1_weighted": f1,
                    "training_time_seconds": training_time,
                },
                f,
                indent=4
            )

        mlflow.log_artifact(metrics_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(3)
        )

    print("Training selesai menggunakan MLflow Project.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
