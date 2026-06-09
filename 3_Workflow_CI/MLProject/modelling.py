import os
import json
from datetime import datetime

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


def load_dataset():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def main():
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("Titanic_MLProject_CI")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_dataset()

    with mlflow.start_run(run_name="Titanic_MLProject_Retrain"):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        params = model.get_params()
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        model_path = os.path.join(ARTIFACT_DIR, "titanic_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        feature_names_path = os.path.join(ARTIFACT_DIR, "feature_names.json")
        with open(feature_names_path, "w", encoding="utf-8") as f:
            json.dump(list(X_train.columns), f, indent=4)
        mlflow.log_artifact(feature_names_path)

        run_info_path = os.path.join(ARTIFACT_DIR, "run_info.json")
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump({"created_at": datetime.now().isoformat(), "metrics": metrics}, f, indent=4)
        mlflow.log_artifact(run_info_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(3),
        )

        print("Training via MLProject selesai.")
        print(metrics)


if __name__ == "__main__":
    main()
