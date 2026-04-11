"""
train_resnet.py
---------------
Train a CNN classifier (ResNet18 / ResNet50 / DenseNet121) for binary
pneumonia detection on chest X-rays from the RSNA dataset.

Key features:
    - Transfer learning from ImageNet-pretrained backbones
    - Weighted cross-entropy loss to handle class imbalance
    - Mild data augmentation (random flip + small rotation) on training set only
    - ReduceLROnPlateau scheduler + early stopping on validation loss
    - Saves best checkpoint, loss/accuracy curves, and per-epoch metrics

Usage:
    python src/classification/train_resnet.py \
        --model_name resnet50 --epochs 25 --batch_size 16 --lr 1e-4
"""

import argparse
import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import PneumoniaDataset


# ------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------
@torch.no_grad()
def compute_confusion_counts(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix counts on a loader.

    Returns:
        TP, FP, TN, FN for the positive class = 1 (pneumonia).
    """
    model.eval()

    tp = fp = tn = fn = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)            # shape [B, 2]
        preds = torch.argmax(logits, dim=1)

        tp += int(((preds == 1) & (labels == 1)).sum().item())
        fp += int(((preds == 1) & (labels == 0)).sum().item())
        tn += int(((preds == 0) & (labels == 0)).sum().item())
        fn += int(((preds == 0) & (labels == 1)).sum().item())

    return tp, fp, tn, fn


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """
    Compute common binary classification metrics from TP/FP/TN/FN.
    Positive class = 1 (pneumonia).
    """
    eps = 1e-12

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / (tp + fp + eps)
    recall_sensitivity = tp / (tp + fn + eps)  # sensitivity / TPR
    specificity = tn / (tn + fp + eps)         # TNR
    f1 = 2 * precision * recall_sensitivity / (precision + recall_sensitivity + eps)
    balanced_acc = 0.5 * (recall_sensitivity + specificity)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall_sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
    }


def compute_train_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute plain accuracy on a loader.
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    return float(correct / max(total, 1))


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
def get_transforms(image_size: int = 224):
    """
    Define train vs validation transforms.

    Notes:
    - Keep augmentation mild for medical images.
    - Apply augmentation ONLY on the training set.
    - Use the same normalization for train and validation.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            # We convert grayscale -> RGB in the dataset (3 channels),
            # so we normalize with 3-channel stats.
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, val_transform


def get_dataloaders(
    labels_csv: str,
    train_split: str,
    val_split: str,
    base_dir: str = ".",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 224,
):
    """
    Create train and validation loaders using:
    - labels_csv: image_path,label
    - train_split / val_split: lists of image paths (one per line)

    Returns:
        train_loader, val_loader, class_weights (tensor length 2), class_counts (dict)
    """
    train_transform, val_transform = get_transforms(image_size=image_size)

    train_dataset = PneumoniaDataset(
        labels_csv=labels_csv,
        split_file=train_split,
        base_dir=base_dir,
        transform=train_transform,
    )

    val_dataset = PneumoniaDataset(
        labels_csv=labels_csv,
        split_file=val_split,
        base_dir=base_dir,
        transform=val_transform,
    )

    # Compute class weights on TRAIN only (to handle imbalance).
    # class_weights[c] ~ inverse frequency
    counts = train_dataset.df["label"].value_counts().sort_index()
    # Ensure both classes exist in the mapping (0 and 1)
    count_0 = int(counts.get(0, 0))
    count_1 = int(counts.get(1, 0))
    if count_0 == 0 or count_1 == 0:
        raise RuntimeError(
            f"Class count issue in train split. count_0={count_0}, count_1={count_1}. "
            "Check your splits/labels."
        )

    total = count_0 + count_1
    # Inverse frequency style: total / (num_classes * count)
    w0 = total / (2 * count_0)
    w1 = total / (2 * count_1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train samples: {len(train_dataset)}  (normal={count_0}, pneumonia={count_1})")
    print(f"[INFO] Val samples:   {len(val_dataset)}")

    class_counts = {"normal": count_0, "pneumonia": count_1}

    return train_loader, val_loader, class_weights, class_counts


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_model(model_name: str = "resnet18") -> nn.Module:
    """
    Build a pretrained model and replace the final layer for 2 classes.
    Supported: resnet18, resnet50, densenet121.
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model

    # default resnet18
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model



# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_resnet(
    labels_csv: str,
    train_split: str,
    val_split: str,
    base_dir: str = ".",
    model_name: str = "resnet18",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    num_workers: int = 4,
    early_stopping_patience: int = 5,
    use_weighted_loss: bool = True,
):
    """
    Train ResNet classifier with:
    - train-only augmentation
    - weighted cross entropy (class imbalance)
    - ReduceLROnPlateau scheduler
    - early stopping on validation loss
    - best checkpoint saving
    - basic medical-relevant metrics (sensitivity/specificity/f1)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Data
    train_loader, val_loader, class_weights, class_counts = get_dataloaders(
        labels_csv=labels_csv,
        train_split=train_split,
        val_split=val_split,
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    # Model
    model = build_model(model_name=model_name).to(device)

    # Loss (weighted)
    class_weights = class_weights.to(device)
    if use_weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(
            f"[INFO] Using weighted cross entropy with class weights: "
            f"normal={class_weights[0].item():.4f}, pneumonia={class_weights[1].item():.4f}"
        )
    else:
        criterion = nn.CrossEntropyLoss()
        print("[INFO] Using standard cross entropy (no class weighting).")

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Logging
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_metrics_history = []  # dict per epoch

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1
    epochs_no_improve = 0

    print("[INFO] Starting training...")

    for epoch in range(epochs):
        # ---------------------- TRAIN ----------------------
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        train_loss = running_loss / max(len(train_loader), 1)
        train_losses.append(train_loss)

        train_acc = compute_train_accuracy(model, train_loader, device)
        train_accuracies.append(train_acc)

        # ---------------------- VAL ----------------------
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_running_loss += float(loss.item())

        val_loss = val_running_loss / max(len(val_loader), 1)
        val_losses.append(val_loss)

        # Confusion-based metrics on validation set
        tp, fp, tn, fn = compute_confusion_counts(model, val_loader, device)
        m = metrics_from_counts(tp, fp, tn, fn)
        val_metrics_history.append(m)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
            f"TrainAcc={train_acc:.4f}  ValAcc={m['accuracy']:.4f}  "
            f"BalAcc={m['balanced_accuracy']:.4f}  "
            f"Sens={m['recall_sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
            f"F1={m['f1']:.4f}  LR={current_lr:.6f}"
        )

        # Early stopping on validation loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"[INFO] Early stopping: no val loss improvement for "
                    f"{early_stopping_patience} epochs."
                )
                break

    # ---------------------- Save best model ----------------------
    Path("models").mkdir(parents=True, exist_ok=True)
    save_path = Path("models") / f"{model_name}_pneumonia_classifier_best.pth"

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Best model saved to: {save_path}")
    if best_epoch >= 0:
        print(f"[INFO] Best epoch: {best_epoch+1} | Best Val Loss: {best_val_loss:.4f}")



    

    # ---------------------- Save plots + metrics summary ----------------------
    # IMPORTANT: save each run in its own folder (no overwrite)
    results_root = Path("results/classification")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    weighted_tag = "weighted" if use_weighted_loss else "unweighted"
    run_name = f"{model_name}_{weighted_tag}_ep{epochs}_bs{batch_size}_lr{lr}_wd{weight_decay}_img{image_size}_pat{early_stopping_patience}_{timestamp}"
    results_dir = results_root / model_name / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    run_title = f"{model_name} | {weighted_tag} | ep={epochs} bs={batch_size} lr={lr} wd={weight_decay} img={image_size} pat={early_stopping_patience}"

    # Loss curves
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    if best_epoch >= 0:
        plt.axvline(best_epoch + 1, linestyle="--", label=f"Best epoch {best_epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training / Validation Loss\n{run_title}")
    plt.legend()
    plt.savefig(results_dir / "loss_curves.png", bbox_inches="tight")
    plt.close()

    # Accuracy curves
    val_acc = [d["accuracy"] for d in val_metrics_history]

    plt.figure()
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    if best_epoch >= 0:
        plt.axvline(best_epoch + 1, linestyle="--", label=f"Best epoch {best_epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training / Validation Accuracy\n{run_title}")
    plt.legend()
    plt.savefig(results_dir / "accuracy_curves.png", bbox_inches="tight")
    plt.close()

    # Validation metrics curves
    val_acc = [d["accuracy"] for d in val_metrics_history]
    val_bal = [d["balanced_accuracy"] for d in val_metrics_history]
    val_sens = [d["recall_sensitivity"] for d in val_metrics_history]

    plt.figure()
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.plot(epochs_range, val_bal, label="Val Balanced Acc")
    plt.plot(epochs_range, val_sens, label="Val Sensitivity")
    if best_epoch >= 0:
        plt.axvline(best_epoch + 1, linestyle="--", label=f"Best epoch {best_epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"Validation Metrics\n{run_title}")
    plt.legend()
    plt.savefig(results_dir / "val_metrics.png", bbox_inches="tight")
    plt.close()

    # Save final metrics (best epoch)
    if best_epoch >= 0 and best_epoch < len(val_metrics_history):
        best_metrics = val_metrics_history[best_epoch]
    else:
        best_metrics = val_metrics_history[-1] if val_metrics_history else {}

    metrics_path = results_dir / "best_val_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"model_name={model_name}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"lr={lr}\n")
        f.write(f"weight_decay={weight_decay}\n")
        f.write(f"image_size={image_size}\n")
        f.write(f"early_stopping_patience={early_stopping_patience}\n")
        f.write(f"use_weighted_loss={use_weighted_loss}\n")
        f.write(f"class_weight_normal={class_weights[0].item():.6f}\n")
        f.write(f"class_weight_pneumonia={class_weights[1].item():.6f}\n")
        f.write(f"train_count_normal={class_counts['normal']}\n")
        f.write(f"train_count_pneumonia={class_counts['pneumonia']}\n")
        f.write(f"best_epoch={best_epoch+1 if best_epoch>=0 else 'NA'}\n")
        f.write(f"best_val_loss={best_val_loss:.6f}\n")
        for k, v in best_metrics.items():
            f.write(f"{k}={v:.6f}\n")

    print(f"[INFO] Saved run artifacts to: {results_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet pneumonia classifier (RSNA).")
    parser.add_argument("--labels_csv", type=str, default="data/classification_labels.csv")
    parser.add_argument("--train_split", type=str, default="data/splits/train.txt")
    parser.add_argument("--val_split", type=str, default="data/splits/val.txt")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--no_weighted_loss", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_resnet(
        labels_csv=args.labels_csv,
        train_split=args.train_split,
        val_split=args.val_split,
        base_dir=args.base_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        use_weighted_loss=not args.no_weighted_loss,
    )
