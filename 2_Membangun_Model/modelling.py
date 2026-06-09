import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRAIN_PATH = os.path.join("namadataset_preprocessing", "train.csv")
TEST_PATH = os.path.join("namadataset_preprocessing", "test.csv")
TARGET_COLUMN = "Survived"


def load_dataset():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def main():
    mlflow.set_experiment("Titanic_Basic_Autolog")
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_dataset()

    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    main()
