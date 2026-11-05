"""Evaluate the saved model on the held-out stratified test partition."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from src.data import load_dataset
from src.model import DateFruitANN

ROOT = Path(__file__).resolve().parents[1]


def evaluate() -> dict:
    torch.set_num_threads(1)
    model_path = ROOT / "models" / "best_model.pt"
    scaler_path = ROOT / "models" / "scaler.joblib"
    encoder_path = ROOT / "models" / "label_encoder.joblib"
    for path in (model_path, scaler_path, encoder_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact: {path}. Run `python -m src.train` first.")
    frame = load_dataset(ROOT / "data" / "DateFruit_Dataset.csv")
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    labels = encoder.transform(frame["Class"].astype(str))
    raw_features = frame.drop(columns="Class").to_numpy(dtype=np.float32)
    x_train, x_temp, y_train, y_temp = train_test_split(
        raw_features, labels, test_size=0.30, random_state=42, stratify=labels
    )
    _, x_test, _, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    x_test = scaler.transform(x_test).astype(np.float32)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if encoder.classes_.tolist() != checkpoint["class_names"]:
        raise ValueError("Saved label encoder does not match checkpoint class mapping.")
    model = DateFruitANN(checkpoint["input_features"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(x_test)).argmax(1).numpy()
    class_ids = list(range(len(encoder.classes_)))
    report = classification_report(y_test, predicted, labels=class_ids,
        target_names=encoder.classes_.tolist(), output_dict=True, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_precision": float(precision), "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=class_ids).tolist(),
        "classification_report": report,
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
