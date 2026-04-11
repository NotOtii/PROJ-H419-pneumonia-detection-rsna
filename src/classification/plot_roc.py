"""
plot_roc.py
-----------
Plot the ROC curve and compute AUC for a trained pneumonia classifier.

The ROC curve is computed manually (no sklearn dependency) by sweeping
over all unique predicted probability thresholds. AUC is computed via
the trapezoidal rule.

Outputs:
    results/classification/<model>/<run>/roc/roc_curve_test.png
    results/classification/<model>/<run>/roc/roc_points.csv
    results/classification/<model>/<run>/roc/roc_summary.json

Usage:
    python src/classification/plot_roc.py \
        --model_name resnet50 \
        --checkpoint models/resnet50_FINAL.pth \
        --batch_size 64
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import PneumoniaDataset


def build_model(model_name: str):
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def compute_roc_curve_manual(y_true, y_prob):
    """
    Manual ROC curve computation without sklearn.
    Returns fpr, tpr, thresholds, auc.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.unique(y_prob)[::-1]
    thresholds = np.concatenate(([1.0 + 1e-8], thresholds, [-1e-8]))

    tpr_list = []
    fpr_list = []

    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # sort by FPR before trapezoidal AUC
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    thresholds = thresholds[order]

    auc = np.trapz(tpr, fpr)

    return fpr, tpr, thresholds, float(auc)


def main():
    parser = argparse.ArgumentParser(description="Plot ROC curve for pneumonia classifier.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default="data/classification_labels.csv")
    parser.add_argument("--test_split", type=str, default="data/splits/test.txt")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--results_root", type=str, default="results/classification")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = PneumoniaDataset(
        labels_csv=args.labels_csv,
        split_file=args.test_split,
        base_dir=args.base_dir,
        transform=test_transform,
        return_path=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model_name).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    all_true = []
    all_prob = []
    all_paths = []

    with torch.inference_mode():
        for images, labels, paths in test_loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # pneumonia probability

            all_true.append(labels.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_paths.extend(list(paths))

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)

    fpr, tpr, thresholds, auc = compute_roc_curve_manual(y_true, y_prob)

    ckpt_stem = Path(args.checkpoint).stem
    parent_run_name = args.run_name or ckpt_stem

    out_dir = Path(args.results_root) / args.model_name / parent_run_name / "roc"
    out_dir.mkdir(parents=True, exist_ok=True)

    roc_png = out_dir / "roc_curve_test.png"
    roc_csv = out_dir / "roc_points.csv"
    roc_json = out_dir / "roc_summary.json"

    # Plot ROC
    plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curve - {args.model_name}")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_png, dpi=220, bbox_inches="tight")
    plt.close()

    # Save ROC points CSV
    with roc_csv.open("w", encoding="utf-8") as f:
        f.write("threshold,fpr,tpr\n")
        for th, x, y in zip(thresholds, fpr, tpr):
            f.write(f"{float(th)},{float(x)},{float(y)}\n")

    # Save summary JSON
    summary = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "labels_csv": args.labels_csv,
        "test_split": args.test_split,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_test_samples": int(len(y_true)),
        "roc_auc": float(auc),
        "roc_curve_path": str(roc_png.as_posix()),
        "roc_points_csv": str(roc_csv.as_posix()),
    }

    with roc_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] ROC-AUC: {auc:.4f}")
    print(f"[INFO] ROC curve saved to: {roc_png}")
    print(f"[INFO] ROC points saved to: {roc_csv}")
    print(f"[INFO] Summary saved to: {roc_json}")


if __name__ == "__main__":
    main()