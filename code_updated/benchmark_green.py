"""
benchmark_green.py  -  Task 2 (Green Initiative) efficiency runner.

Separate from the Task 1 benchmark.py. It trains every model (AlexNet, VGG16,
ResNet18, GreenResNet) on every dataset and logs, per run:
    (i)   total training runtime
    (ii)  inference latency per sample
    (iii) peak memory during training and during inference
plus the parameter count, so the green model can be compared head-to-head.

Reads dataset/training settings from the same config.json (DATASETS, DATA_PATH,
BATCH_SIZE, EPOCHS, ...), but defines its OWN model list below so Task 1's
config does not need to change.
"""
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from data import get_loaders
import models
from models_green import GreenResNet
from fit import Trainer

# Task 2 compares the baseline against its green version only.
PROFILE_MODELS = ["AlexNet", "VGG16", "ResNet18", "GreenResNet"]


def get_model_class(name):
    """Look the model up in models.py first, then the green module."""
    if hasattr(models, name):
        return getattr(models, name)
    return {"GreenResNet": GreenResNet}[name]


def peak_mem_mb(device):
    """Peak GPU memory since the last reset, in MB. CPU has no counter -> 0.0."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def reset_mem(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def sync(device):
    """Wait for GPU work to finish before reading the clock (else we time queuing)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, test_loader, device):
    """Average inference time per sample (ms) + peak inference memory.
    One warm-up batch is discarded, then the whole test set is timed."""
    model.eval()
    n_samples = 0
    with torch.no_grad():
        for images, _ in test_loader:          # warm-up (not timed)
            model(images.to(device))
            break

        reset_mem(device)
        sync(device)
        start = time.perf_counter()
        for images, _ in test_loader:
            model(images.to(device))
            n_samples += images.size(0)
        sync(device)
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / n_samples) * 1000.0
    return latency_ms, peak_mem_mb(device)


def test_accuracy(model, test_loader, device):
    """Test accuracy + macro F1 (enough for the green comparison)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            all_preds += predicted.cpu().tolist()
            all_labels += labels.cpu().tolist()

    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    acc = 100 * correct / len(all_labels)
    f1 = 100 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


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

        for model_name in PROFILE_MODELS:
            print(f"\n=== {model_name} on {data} ===")

            model_class = get_model_class(model_name)
            model = model_class(in_channels=in_channels, num_classes=num_classes,
                                drop_rate=config["DROP_RATE"],
                                activation_str=config.get("ACTIVATION")).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
            trainer = Trainer(model, criterion, optimizer, device)

            # (i) training runtime + (iii) training peak memory
            reset_mem(device)
            sync(device)
            t0 = time.perf_counter()
            trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
            sync(device)
            train_time = time.perf_counter() - t0
            train_mem = peak_mem_mb(device)

            # (ii) inference latency per sample + inference peak memory
            latency_ms, infer_mem = measure_latency(model, test_loader, device)

            acc, f1 = test_accuracy(model, test_loader, device)
            n_params = count_params(model)

            results.append((data, model_name, acc, f1, n_params,
                            train_time, latency_ms, train_mem, infer_mem))
            print(f"acc:{acc:.2f}%  f1:{f1:.2f}%  params:{n_params:,}  "
                  f"train:{train_time:.1f}s  latency:{latency_ms:.3f}ms/img  "
                  f"mem(train/infer):{train_mem:.0f}/{infer_mem:.0f}MB")

    print("\n\n===== TASK 2: GREEN EFFICIENCY MATRIX =====")
    print(f"{'dataset':<10}{'model':<13}{'acc':>7}{'f1':>7}{'params':>12}"
          f"{'train_s':>10}{'ms/img':>10}{'tr_MB':>9}{'inf_MB':>9}")
    for data, model_name, acc, f1, n_params, train_time, latency_ms, train_mem, infer_mem in results:
        print(f"{data:<10}{model_name:<13}{acc:>6.1f}%{f1:>6.1f}%{n_params:>12,}"
              f"{train_time:>10.1f}{latency_ms:>10.3f}{train_mem:>9.0f}{infer_mem:>9.0f}")


if __name__ == "__main__":
    main()