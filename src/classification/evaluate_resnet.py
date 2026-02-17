import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import sys
from pathlib import Path
print("[DEBUG] evaluate_resnet.py started")
sys.path.append(str(Path(__file__).resolve().parent))


from dataset import PneumoniaDataset


def compute_metrics(y_true, y_pred, y_prob):
    # Confusion components (positive class = 1)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    eps = 1e-12
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / (tp + fn + eps)  # recall pneumonia
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
    balanced_acc = 0.5 * (sensitivity + specificity)

    # ROC-AUC (manual, no sklearn)
    # Rank-based AUC (equivalent to Mann–Whitney U)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        # Compute probability that a random positive has higher score than a random negative
        auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "roc_auc": auc,
    }


def plot_confusion_matrix(tp, fp, tn, fn, save_path):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Normal(0)", "Pneumonia(1)"])
    plt.yticks([0, 1], ["Normal(0)", "Pneumonia(1)"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def build_model(model_name: str):
    model_name = model_name.lower().strip()
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--checkpoint", type=str, default="models/resnet50_pneumonia_classifier.pth")
    parser.add_argument("--labels_csv", type=str, default="data/classification_labels.csv")
    parser.add_argument("--test_split", type=str, default="data/splits/test.txt")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Same deterministic preprocessing as validation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485],
                             std=[0.229, 0.229, 0.229]),
    ])

    test_dataset = PneumoniaDataset(
        labels_csv=args.labels_csv,
        split_file=args.test_split,
        base_dir=args.base_dir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(args.model_name).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)

            preds = torch.argmax(logits, dim=1)

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("[INFO] --- TEST METRICS ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    results_dir = Path("results/classification")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix figure
    cm_path = results_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"], cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    # Save metrics to file
    out_txt = results_dir / "test_metrics.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")
    print(f"[INFO] Metrics saved to {out_txt}")


if __name__ == "__main__":
    main()
