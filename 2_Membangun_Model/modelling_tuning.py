import os
import json
import time
from datetime import datetime

import joblib
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import GridSearchCV


# ============================================================
# KONFIGURASI PATH DAN IDENTITAS PROJECT
# ============================================================

TRAIN_PATH = os.path.join("namadataset_preprocessing", "train.csv")
TEST_PATH = os.path.join("namadataset_preprocessing", "test.csv")
TARGET_COLUMN = "Survived"
ARTIFACT_DIR = "artifacts"

DAGSHUB_REPO_OWNER = "msidiqfirdauss"
DAGSHUB_REPO_NAME = "Eksperimen_SML_MSidiqFirdaus"
MLFLOW_EXPERIMENT_NAME = "Titanic_Advanced_Manual_Logging"


# ============================================================
# KONFIGURASI MLFLOW + DAGSHUB
# ============================================================

def configure_mlflow():
    """
    Menghubungkan MLflow dengan DagsHub.

    Catatan:
    - Untuk lokal, dagshub.init biasanya akan meminta login melalui browser.
    - Setelah login berhasil, MLflow run akan masuk ke tab Experiments di DagsHub.
    """

    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("MLflow berhasil dikonfigurasi dengan DagsHub.")
    print(f"Repo Owner : {DAGSHUB_REPO_OWNER}")
    print(f"Repo Name  : {DAGSHUB_REPO_NAME}")
    print(f"Experiment : {MLFLOW_EXPERIMENT_NAME}")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Membaca dataset hasil preprocessing.
    Dataset yang digunakan harus sudah berbentuk train.csv dan test.csv.
    """

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"File train tidak ditemukan: {TRAIN_PATH}")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"File test tidak ditemukan: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan di train.csv")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan di test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


# ============================================================
# MEMBUAT ARTIFAK TAMBAHAN
# ============================================================

def create_artifacts(model, X_test, y_test, y_pred, metrics):
    """
    Membuat artifak tambahan untuk memenuhi kriteria Advanced:
    - classification_report.txt
    - confusion_matrix.png
    - feature_importance.png
    - feature_importance.csv
    - model_info.json
    - titanic_model.pkl
    - feature_names.json
    """

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    artifact_paths = []

    # 1. Classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    artifact_paths.append(report_path)

    # 2. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.title("Titanic Confusion Matrix")

    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    artifact_paths.append(cm_path)

    # 3. Feature importance
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        feature_importance_csv_path = os.path.join(
            ARTIFACT_DIR,
            "feature_importance.csv"
        )

        importance_df.to_csv(feature_importance_csv_path, index=False)
        artifact_paths.append(feature_importance_csv_path)

        plt.figure(figsize=(10, 5))
        importance_df.head(15).plot(
            kind="bar",
            x="feature",
            y="importance",
            legend=False
        )
        plt.title("Top Feature Importance")
        plt.ylabel("Importance")
        plt.tight_layout()

        feature_importance_png_path = os.path.join(
            ARTIFACT_DIR,
            "feature_importance.png"
        )

        plt.savefig(feature_importance_png_path, bbox_inches="tight")
        plt.close()

        artifact_paths.append(feature_importance_png_path)

    # 4. Model pickle untuk kebutuhan monitoring / inference lokal
    model_path = os.path.join(ARTIFACT_DIR, "titanic_model.pkl")
    joblib.dump(model, model_path)
    artifact_paths.append(model_path)

    # 5. Feature names
    feature_names_path = os.path.join(ARTIFACT_DIR, "feature_names.json")

    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(list(X_test.columns), f, indent=4)

    artifact_paths.append(feature_names_path)

    # 6. Model info
    model_info = {
        "model_name": "RandomForestClassifier",
        "dataset": "Titanic Dataset",
        "target_column": TARGET_COLUMN,
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "features": list(X_test.columns),
        "dagshub_repo_owner": DAGSHUB_REPO_OWNER,
        "dagshub_repo_name": DAGSHUB_REPO_NAME,
        "mlflow_experiment_name": MLFLOW_EXPERIMENT_NAME,
    }

    model_info_path = os.path.join(ARTIFACT_DIR, "model_info.json")

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)

    artifact_paths.append(model_info_path)

    return artifact_paths


# ============================================================
# TRAINING + TUNING + MANUAL LOGGING
# ============================================================

def main():
    configure_mlflow()

    X_train, X_test, y_train, y_test = load_dataset()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    base_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # Untuk ROC-AUC binary classification
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = 0.0

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "roc_auc": roc_auc,
        "best_cv_score": grid_search.best_score_,
        "training_time_seconds": training_time,
    }

    artifact_paths = create_artifacts(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        metrics=metrics
    )

    with mlflow.start_run(run_name="Titanic_RandomForest_Tuning_DagsHub_Manual"):
        # ====================================================
        # MANUAL LOGGING PARAMETER
        # ====================================================

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv", 5)
        mlflow.log_param("scoring", "f1_weighted")
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("dataset", "Titanic Dataset")
        mlflow.log_param("tracking", "DagsHub MLflow")
        mlflow.log_param("repo_owner", DAGSHUB_REPO_OWNER)
        mlflow.log_param("repo_name", DAGSHUB_REPO_NAME)

        # ====================================================
        # MANUAL LOGGING METRIC
        # ====================================================

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))

        # ====================================================
        # MANUAL LOGGING ARTIFAK TAMBAHAN
        # ====================================================

        for artifact_path in artifact_paths:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

        # ====================================================
        # MANUAL LOGGING MODEL
        # ====================================================

        input_example = X_test.head(3)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example
        )

    print("\nTraining tuning selesai dan berhasil dikirim ke DagsHub MLflow.")
    print("Best params:")
    print(grid_search.best_params_)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()