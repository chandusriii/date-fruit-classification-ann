"""Train the date-fruit ANN and save the best validation checkpoint."""
import argparse
import json
import random
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.data import load_dataset, prepare_data
from src.model import DateFruitANN

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(x, y, batch_size: int, shuffle: bool):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle)


def train(args) -> dict:
    set_seed(args.seed)
    torch.set_num_threads(1)
    frame = load_dataset(ROOT / "data" / "DateFruit_Dataset.csv")
    data = prepare_data(frame, test_size=args.test_size,
                        validation_size=args.validation_size, seed=args.seed)
    train_loader = make_loader(data["x_train"], data["y_train"], args.batch_size, True)
    val_loader = make_loader(data["x_validation"], data["y_validation"], args.batch_size, False)
    model = DateFruitANN(len(data["feature_names"]), len(data["class_names"]))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model_dir, results_dir = ROOT / "models", ROOT / "results"
    model_dir.mkdir(exist_ok=True); results_dir.mkdir(exist_ok=True)
    best_accuracy, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_loss = 0.0; seen = 0
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item() * len(labels); seen += len(labels)
        model.eval(); val_loss = 0.0; correct = total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                logits = model(features)
                val_loss += criterion(logits, labels).item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item(); total += len(labels)
        train_loss = epoch_loss / max(seen, 1)
        val_loss /= max(total, 1); val_accuracy = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": val_loss, "validation_accuracy": val_accuracy})
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.5f} | "
              f"val loss {val_loss:.5f} | val accuracy {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_features": len(data["feature_names"]),
                "num_classes": len(data["class_names"]),
                "feature_names": data["feature_names"],
                "class_names": data["class_names"],
                "best_validation_accuracy": best_accuracy,
                "best_epoch": best_epoch, "seed": args.seed,
            }, model_dir / "best_model.pt")
    joblib.dump(data["scaler"], model_dir / "scaler.joblib")
    joblib.dump(data["label_encoder"], model_dir / "label_encoder.joblib")
    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot([h["train_loss"] for h in history], label="Training loss")
    plt.plot([h["validation_loss"] for h in history], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
    plt.title("Date Fruit ANN Training"); plt.legend(); plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=150); plt.close()
    summary = {
        "samples": len(frame), "features": len(data["feature_names"]),
        "classes": data["class_names"],
        "split_sizes": {"train": len(data["y_train"]),
                        "validation": len(data["y_validation"]),
                        "test": len(data["y_test"])},
        "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
        "epochs_requested": args.epochs, "seed": args.seed,
    }
    (results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining summary:\n" + json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
