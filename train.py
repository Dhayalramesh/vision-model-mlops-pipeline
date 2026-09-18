"""
Standalone training script — same logic as the Colab notebook, runnable via:
    python train.py

Used by GitHub Actions CI to verify the pipeline still runs end-to-end on every push.
"""

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("image_classifier_experiments")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_data(train_size=2000, test_size=500):
    """Small subset by default so CI runs fast; pass larger sizes for a real training run."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    full_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return Subset(full_train, range(train_size)), Subset(full_test, range(test_size))


def train_and_log(train_set, test_set, lr, epochs, batch_size, run_name):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"lr": lr, "epochs": epochs, "batch_size": batch_size})

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        mlflow.log_metric("val_accuracy", val_acc)

        return val_acc, model


def main():
    # Use --fast for a tiny CI smoke test; omit for a fuller local/Colab run
    fast_mode = "--fast" in sys.argv
    train_size, test_size = (500, 200) if fast_mode else (8000, 2000)
    epochs = 1 if fast_mode else 3

    train_set, test_set = get_data(train_size, test_size)

    configs = [
        {"lr": 0.001, "epochs": epochs, "batch_size": 64, "run_name": "run_lr001"},
        {"lr": 0.0005, "epochs": epochs, "batch_size": 64, "run_name": "run_lr0005"},
    ]

    results = []
    for cfg in configs:
        val_acc, model = train_and_log(train_set, test_set, **cfg)
        results.append({"run_name": cfg["run_name"], "val_acc": val_acc, "model": model})
        print(f"{cfg['run_name']}: val_acc={val_acc:.4f}")

    best = max(results, key=lambda r: r["val_acc"])
    print(f"\nBest run: {best['run_name']} (val_acc={best['val_acc']:.4f})")

    runs_df = mlflow.search_runs(experiment_names=["image_classifier_experiments"])
    runs_df = runs_df[["tags.mlflow.runName", "params.lr", "params.epochs",
                        "params.batch_size", "metrics.val_accuracy", "metrics.train_loss"]]
    runs_df.columns = ["run_name", "lr", "epochs", "batch_size", "val_accuracy", "train_loss"]
    runs_df.to_csv("runs_summary.csv", index=False)

    if not fast_mode:
        torch.save(best["model"].state_dict(), "best_model.pth")
        print("Saved best_model.pth and runs_summary.csv")


if __name__ == "__main__":
    main()
