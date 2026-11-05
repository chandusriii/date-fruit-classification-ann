"""Data loading, validation, stratified splitting, and scaling."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COLUMN = "Class"


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COLUMN}'.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")
    features = frame.drop(columns=TARGET_COLUMN)
    if features.isna().any().any():
        raise ValueError("Feature columns contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All feature columns must be numeric.")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Features contain non-finite values.")
    return frame


def prepare_data(frame: pd.DataFrame, test_size: float = 0.15,
                 validation_size: float = 0.15, seed: int = 42):
    """Return scaled arrays and fitted encoders using training-only fit."""
    X = frame.drop(columns=TARGET_COLUMN).to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[TARGET_COLUMN].astype(str)).astype(np.int64)
    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + validation_size,
        random_state=seed, stratify=y
    )
    relative_test = test_size / (test_size + validation_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=relative_test, random_state=seed, stratify=y_temp
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return {
        "x_train": x_train, "x_validation": x_val, "x_test": x_test,
        "y_train": y_train, "y_validation": y_val, "y_test": y_test,
        "scaler": scaler, "label_encoder": encoder,
        "feature_names": frame.drop(columns=TARGET_COLUMN).columns.tolist(),
        "class_names": encoder.classes_.tolist(),
    }
