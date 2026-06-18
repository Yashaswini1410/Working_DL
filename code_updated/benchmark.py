"""
benchmark.py - trains every model on every dataset, then tests each one.

Same style as train.py and fit.py - just wrapped in a loop so it does
all 12 combinations and prints a results table at the end.
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

from data import get_loaders
import models
from fit import Trainer


def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds += predicted.cpu().tolist()
            all_labels += labels.cpu().tolist()

    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    acc = 100 * correct / len(all_labels)

    prec = 100 * precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = 100 * recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = 100 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, prec, rec, f1


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    results = []

    for data in config["DATASETS"]:
        train_loader, val_loader, test_loader = get_loaders(
            data=data, data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

        in_channels = train_loader.dataset.tensors[0].shape[1]
        num_classes = int(train_loader.dataset.tensors[1].max()) + 1

        for model_name in config["MODELS"]:
            print(f"\n=== {model_name} on {data} ===")

            model_class = getattr(models, model_name)
            model = model_class(in_channels=in_channels, num_classes=num_classes,
                                drop_rate=config["DROP_RATE"],
                                activation_str=config.get("ACTIVATION")).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

            trainer = Trainer(model, criterion, optimizer, device)
            trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

            acc, prec, rec, f1 = test_model(model, test_loader, device)
            results.append((data, model_name, acc, prec, rec, f1))
            print(f"TEST -> acc:{acc:.2f}%  prec:{prec:.2f}%  rec:{rec:.2f}%  f1:{f1:.2f}%")

    print("\n\n===== FINAL RESULTS =====")
    print(f"{'dataset':<10}{'model':<10}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
    for data, model_name, acc, prec, rec, f1 in results:
        print(f"{data:<10}{model_name:<10}{acc:>7.1f}%{prec:>7.1f}%{rec:>7.1f}%{f1:>7.1f}%")


if __name__ == "__main__":
    main()