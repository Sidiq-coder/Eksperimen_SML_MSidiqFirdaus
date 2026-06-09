import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = os.path.join("..", "namadataset_raw", "titanic.csv")
OUTPUT_DIR = "namadataset_preprocessing"
TARGET_COLUMN = "Survived"
RANDOM_STATE = 42
TEST_SIZE = 0.2

DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def make_one_hot_encoder():
    """Compatibility helper untuk scikit-learn versi lama dan baru."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {path}. "
            "Letakkan Titanic dataset dengan nama titanic.csv pada folder namadataset_raw."
        )
    return pd.read_csv(path)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    # Pastikan kolom target tersedia
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan pada dataset.")

    # Feature engineering sederhana dari dataset Titanic
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Handling missing value
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Drop kolom high-cardinality atau terlalu banyak missing value
    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    return df


def split_dataset(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    available_numeric = [col for col in NUMERIC_FEATURES if col in X_train.columns]
    available_categorical = [col for col in CATEGORICAL_FEATURES if col in X_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), available_numeric),
            ("cat", make_one_hot_encoder(), available_categorical),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())

    train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

    return train_processed_df, test_processed_df, feature_names


def save_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame, metadata: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    metadata_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Data train tersimpan di: {train_path}")
    print(f"Data test tersimpan di: {test_path}")
    print(f"Metadata tersimpan di: {metadata_path}")


def run_preprocessing():
    df_raw = load_data()
    df_clean = clean_and_engineer_features(df_raw)
    X_train, X_test, y_train, y_test = split_dataset(df_clean)
    X_train_processed, X_test_processed, feature_names = preprocess_features(X_train, X_test)

    train_df = X_train_processed.copy()
    train_df[TARGET_COLUMN] = y_train.reset_index(drop=True)

    test_df = X_test_processed.copy()
    test_df[TARGET_COLUMN] = y_test.reset_index(drop=True)

    metadata = {
        "dataset_name": "Titanic Dataset",
        "target_column": TARGET_COLUMN,
        "raw_shape": list(df_raw.shape),
        "clean_shape": list(df_clean.shape),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "drop_columns": DROP_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "processed_feature_names": feature_names,
        "created_at": datetime.now().isoformat(),
    }

    save_outputs(train_df, test_df, metadata)
    return train_df, test_df, metadata


if __name__ == "__main__":
    run_preprocessing()
