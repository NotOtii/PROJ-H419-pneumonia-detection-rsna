"""
evaluate_resnet.py
------------------
Evaluate a trained pneumonia classifier on the held-out test set.

Produces:
    - Confusion matrix plot
    - Probability histogram (per-class distribution of predicted probabilities)
    - Per-image prediction CSV
    - Summary metrics (accuracy, sensitivity, specificity, F1, balanced accuracy, ROC-AUC)
    - Optional export of top false-positive / false-negative images for error analysis

Threshold selection:
    - If --threshold is given, that value is used directly.
    - Otherwise, the script auto-selects the threshold that maximises balanced accuracy
      while meeting --target_sensitivity (default 0.90).

Usage:
    python src/classification/evaluate_resnet.py \
        --model_name resnet50 \
        --checkpoint models/resnet50_FINAL.pth \
        --batch_size 64 --threshold 0.3
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import PneumoniaDataset


def compute_metrics(y_true, y_pred, y_prob):
    """Compute binary classification metrics. Positive class = 1 (pneumonia).
    ROC-AUC is computed via the Mann-Whitney U statistic (no sklearn needed)."""
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

    # ROC-AUC (manual, no sklearn) - rank-based AUC (Mann–Whitney U)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        auc = float(
            (pos[:, None] > neg[None, :]).mean()
            + 0.5 * (pos[:, None] == neg[None, :]).mean()
        )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "roc_auc": auc,
    }


def plot_confusion_matrix(tp, fp, tn, fn, save_path):
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    total = max(cm.sum(), 1.0)
    cm_pct = cm / total

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="RdBu_r")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=18)

    ax.set_title("Confusion Matrix (Test)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xticks([0, 1], ["Normal (0)", "Pneumonia (1)"])
    ax.set_yticks([0, 1], ["Normal (0)", "Pneumonia (1)"])

    max_value = cm.max() if cm.size else 1.0
    threshold = max_value * 0.55

    for (i, j), v in np.ndenumerate(cm):
        text_color = "white" if v >= threshold else "black"
        ax.text(
            j,
            i,
            f"{int(v)}\n({cm_pct[i, j] * 100:.1f}%)",
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
            fontweight="semibold",
        )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.close()


def build_model(model_name: str):
    """
    Build the SAME architecture as in train_resnet.py.
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


def pick_threshold_for_target_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target_sens: float):
    """Find the threshold that maximises balanced accuracy while keeping sensitivity >= target_sens."""
    thresholds = np.linspace(0.0, 1.0, 501)
    best_t = 0.5
    best_bal = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        m = compute_metrics(y_true, y_pred, y_prob)
        if m["sensitivity"] >= target_sens:
            if m["balanced_accuracy"] > best_bal:
                best_bal = m["balanced_accuracy"]
                best_t = t

    return float(best_t), float(best_bal)


def save_top_errors(
    y_true,
    y_pred,
    y_prob,
    paths,
    out_dir,
    max_each=20,
):
    """Save the highest-confidence false positives and false negatives as PNG for review."""
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_sorted = np.argsort(-y_prob)

    saved_fp = 0
    saved_fn = 0

    for i in idx_sorted:
        true = int(y_true[i])
        pred = int(y_pred[i])
        prob = float(y_prob[i])
        p = Path(paths[i])

        try:
            if pred == 1 and true == 0 and saved_fp < max_each:
                out = out_dir / f"FP_prob{prob:.3f}_{p.name}"
                Image.open(p).save(out)
                saved_fp += 1

            if pred == 0 and true == 1 and saved_fn < max_each:
                out = out_dir / f"FN_prob{prob:.3f}_{p.name}"
                Image.open(p).save(out)
                saved_fn += 1

        except Exception as e:
            print(f"[WARNING] Could not save {p}: {e}")

        if saved_fp >= max_each and saved_fn >= max_each:
            break

    print(f"[INFO] Saved errors -> {out_dir} (FP={saved_fp}, FN={saved_fn})")


def save_probability_histogram(y_true, y_prob, save_path):
    plt.figure(figsize=(9, 5.5))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.65, label="True Normal (0)")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.65, label="True Pneumonia (1)")
    plt.xlabel("Predicted probability for Pneumonia (class 1)")
    plt.ylabel("Count")
    plt.title("Probability Distribution on Test Set")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.close()


def save_metrics_text(path: Path, metrics: dict):
    with path.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")


def save_metrics_json(path: Path, metrics: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_predictions_csv(path: Path, y_true, y_pred, y_prob, all_paths):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "y_true", "y_pred", "prob_pneumonia"])
        for img_path, true, pred, prob in zip(all_paths, y_true, y_pred, y_prob):
            writer.writerow([img_path, int(true), int(pred), float(prob)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--checkpoint", type=str, default="models/resnet50_pneumonia_classifier_best.pth")
    parser.add_argument("--labels_csv", type=str, default="data/classification_labels.csv")
    parser.add_argument("--test_split", type=str, default="data/splits/test.txt")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--results_root", type=str, default="results/classification")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target_sensitivity", type=float, default=0.90)

    parser.add_argument("--save_errors", action="store_true")
    parser.add_argument("--max_errors_each", type=int, default=20)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model_name).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_true = []
    all_prob = []
    all_paths = []

    with torch.inference_mode():
        for images, labels, paths in test_loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_true.append(labels.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_paths.extend(list(paths))

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)

    if args.threshold is None:
        chosen_t, best_bal = pick_threshold_for_target_sensitivity(
            y_true=y_true, y_prob=y_prob, target_sens=args.target_sensitivity
        )
        print(f"[INFO] Auto threshold picked: {chosen_t:.3f} "
              f"(target_sensitivity={args.target_sensitivity}, best_bal_acc={best_bal:.4f})")
    else:
        chosen_t = float(args.threshold)
        print(f"[INFO] Using user threshold: {chosen_t:.3f}")

    y_pred = (y_prob >= chosen_t).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = float(chosen_t)
    metrics["model_name"] = args.model_name
    metrics["checkpoint"] = args.checkpoint
    metrics["batch_size"] = args.batch_size

    print("[INFO] --- TEST METRICS ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # FIXED INDENTATION BLOCK
    results_root = Path(args.results_root)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_stem = Path(args.checkpoint).stem
    th_str = (
        f"th{chosen_t:.3f}"
        if args.threshold is not None
        else f"autoSens{args.target_sensitivity:.2f}_th{chosen_t:.3f}"
    )

    parent_run_name = args.run_name or ckpt_stem
    results_dir = results_root / args.model_name / parent_run_name / "evaluation" / f"{th_str}__bs{args.batch_size}__{timestamp}"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cm_path = plots_dir / f"confusion_matrix_test_{chosen_t:.3f}.png"
    plot_confusion_matrix(metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"], cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    prob_hist_path = plots_dir / f"probability_histogram_test_{chosen_t:.3f}.png"
    save_probability_histogram(y_true, y_prob, prob_hist_path)
    print(f"[INFO] Probability histogram saved to {prob_hist_path}")

    out_txt = results_dir / f"test_metrics_{chosen_t:.3f}.txt"
    save_metrics_text(out_txt, metrics)
    print(f"[INFO] Metrics saved to {out_txt}")

    out_json = results_dir / f"test_metrics_{chosen_t:.3f}.json"
    save_metrics_json(out_json, metrics)

    predictions_csv = results_dir / f"test_predictions_{chosen_t:.3f}.csv"
    save_predictions_csv(predictions_csv, y_true, y_pred, y_prob, all_paths)

    run_summary = {
        "parent_run_name": parent_run_name,
        "evaluation_dir": str(results_dir.as_posix()),
        "plots_dir": str(plots_dir.as_posix()),
        "confusion_matrix_path": str(cm_path.as_posix()),
        "probability_histogram_path": str(prob_hist_path.as_posix()),
        "predictions_csv": str(predictions_csv.as_posix()),
    }
    save_metrics_json(results_dir / "evaluation_summary.json", run_summary)

    if args.save_errors:
        errors_dir = results_dir / "errors"
        save_top_errors(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            paths=all_paths,
            out_dir=errors_dir,
            max_each=args.max_errors_each,
        )


if __name__ == "__main__":
    main()