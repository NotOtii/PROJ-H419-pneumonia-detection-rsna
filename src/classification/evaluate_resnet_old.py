import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

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
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

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
        pin_memory=True,
    )

    model = build_model(args.model_name).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
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

    print("[INFO] --- TEST METRICS ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # 🔥 FIXED INDENTATION BLOCK
    results_root = Path(args.results_root)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_stem = Path(args.checkpoint).stem
    th_str = (
        f"th{chosen_t:.3f}"
        if args.threshold is not None
        else f"autoSens{args.target_sensitivity:.2f}_th{chosen_t:.3f}"
    )
    run_name = (
        args.run_name
        or f"{args.model_name}__{ckpt_stem}__{th_str}__bs{args.batch_size}__{timestamp}"
    )

    results_dir = results_root / args.model_name / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    cm_path = results_dir / f"confusion_matrix_test_{chosen_t:.3f}.png"
    plot_confusion_matrix(metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"], cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    out_txt = results_dir / f"test_metrics_{chosen_t:.3f}.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")
    print(f"[INFO] Metrics saved to {out_txt}")

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


"""
python src/classification/evaluate_resnet.py --model_name densenet121 --checkpoint models/densenet121_pneumonia_classifier_best.pth --batch_size 64 --save_errors --max_errors_each 10
python src/classification/evaluate_resnet.py --model_name resnet18 --checkpoint models/resnet18_pneumonia_classifier_best.pth --batch_size 64 --save_errors --max_errors_each 10

"""