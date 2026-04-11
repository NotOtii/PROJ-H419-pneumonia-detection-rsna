"""
gradcam_batch.py
----------------
Generate Grad-CAM heatmaps for a batch of correctly classified test images.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions
of the input image that most influenced the model's prediction, providing
visual interpretability for the CNN classifier.

The script:
    1. Scans the test set for True Positive (TP) and True Negative (TN) cases
    2. Ranks them by prediction confidence
    3. Generates per-image outputs: original, heatmap, overlay
    4. Creates a combined TP/TN panel figure for the report

Usage:
    python src/classification/gradcam_batch.py \
        --model_name resnet50 \
        --checkpoint models/resnet50_FINAL.pth \
        --labels_csv data/classification_labels.csv \
        --split_txt data/splits/test.txt \
        --num_tp 5 --num_tn 5
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


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

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_target_layer(model, model_name: str):
    """Return the last convolutional layer for Grad-CAM hook registration."""
    model_name = model_name.lower().strip()

    if model_name in ["resnet18", "resnet50"]:
        return model.layer4[-1]

    if model_name == "densenet121":
        return model.features

    raise ValueError(f"Unsupported model_name: {model_name}")


class GradCAM:
    """Grad-CAM implementation using forward/backward hooks on a target layer.

    Hooks capture the feature-map activations (forward pass) and their gradients
    (backward pass). The class activation map is computed as the ReLU of the
    weighted sum of activations, where weights are the global-average-pooled gradients.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks to capture activations and gradients
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam


def overlay_heatmap_on_image(rgb_img_float, cam, alpha=0.4):
    """Blend a Grad-CAM heatmap onto the original RGB image."""
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = heatmap * alpha + rgb_img_float * (1 - alpha)
    overlay = overlay / np.maximum(overlay.max(), 1e-8)
    overlay = np.uint8(255 * overlay)

    return overlay, np.uint8(255 * heatmap), np.uint8(255 * rgb_img_float)


def save_image(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class SimpleImageDataset(Dataset):
    def __init__(self, labels_csv, split_txt, base_dir=".", image_size=224):
        self.base_dir = Path(base_dir)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        split_paths = []
        with open(split_txt, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    split_paths.append(p.replace("\\", "/"))

        label_map = {}
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames

            # try common column names
            path_col = None
            label_col = None

            for c in cols:
                lc = c.lower()
                if path_col is None and ("path" in lc or "image" in lc or "file" in lc):
                    path_col = c
                if label_col is None and ("label" in lc or "target" in lc or "class" in lc):
                    label_col = c

            if path_col is None or label_col is None:
                raise ValueError(
                    f"Could not detect path/label columns in {labels_csv}. Columns found: {cols}"
                )

            for row in reader:
                rel_path = row[path_col].strip().replace("\\", "/")
                label = int(row[label_col])
                label_map[rel_path] = label

        self.samples = []
        for rel_path in split_paths:
            if rel_path in label_map:
                self.samples.append((rel_path, label_map[rel_path]))
            else:
                # fallback: try normalized relative path
                rp = rel_path.replace("\\", "/")
                if rp in label_map:
                    self.samples.append((rp, label_map[rp]))

        if len(self.samples) == 0:
            raise ValueError("No matching samples found between split file and labels CSV.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        abs_path = self.base_dir / rel_path

        pil_img = Image.open(abs_path).convert("RGB")
        rgb_img = np.array(pil_img.resize((self.image_size, self.image_size))).astype(np.float32) / 255.0
        input_tensor = self.transform(pil_img)

        return input_tensor, label, rgb_img, str(abs_path), rel_path


def rank_candidates(candidates, prefer="high_confidence"):
    """
    candidates: list of dicts containing pneumonia_probability / normal_probability
    """
    if prefer == "high_confidence":
        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)

    if prefer == "low_confidence":
        return sorted(candidates, key=lambda x: x["confidence"])

    return candidates


def make_panel_figure(tp_items, tn_items, out_path, cell_size=224, pad=16, label_h=38):
    """
    Creates a 2-row figure:
      row 1 = TP overlays
      row 2 = TN overlays
    """
    cols = max(len(tp_items), len(tn_items))
    rows = 2

    width = cols * cell_size + (cols + 1) * pad
    height = rows * (cell_size + label_h) + (rows + 1) * pad

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_row(items, row_idx, row_title):
        y0 = pad + row_idx * (cell_size + label_h + pad)

        cv2.putText(
            canvas, row_title, (pad, y0 + 20),
            font, 0.65, (0, 0, 0), 2, cv2.LINE_AA
        )

        for i, item in enumerate(items):
            x = pad + i * (cell_size + pad)
            y = y0 + label_h

            img = item["overlay_img"]
            if img.shape[:2] != (cell_size, cell_size):
                img = cv2.resize(img, (cell_size, cell_size))

            canvas[y:y + cell_size, x:x + cell_size] = img

            short_name = Path(item["rel_path"]).stem[:16]
            prob_txt = f"p={item['pneumonia_probability']:.2f}"

            cv2.putText(canvas, short_name, (x, y - 14), font, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, prob_txt, (x, y + cell_size + 18), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    draw_row(tp_items, 0, "True Positives (Pneumonia correctly detected)")
    draw_row(tn_items, 1, "True Negatives (Normal correctly classified)")

    Image.fromarray(canvas).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate batch Grad-CAM figure for TP and TN cases.")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--split_txt", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_tp", type=int, default=5)
    parser.add_argument("--num_tn", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="results/classification/gradcam_batch")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--prefer", type=str, default="high_confidence", choices=["high_confidence", "low_confidence"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_model(args.model_name).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dataset = SimpleImageDataset(
        labels_csv=args.labels_csv,
        split_txt=args.split_txt,
        base_dir=args.base_dir,
        image_size=args.image_size
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    tp_candidates = []
    tn_candidates = []

    print("[INFO] Scanning dataset for TP / TN candidates...")

    with torch.no_grad():
        for batch in loader:
            input_tensors, labels, rgb_imgs, abs_paths, rel_paths = batch
            input_tensors = input_tensors.to(device)

            logits = model(input_tensors)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            labels_np = labels.numpy()

            for i in range(len(preds)):
                pred = int(preds[i])
                true = int(labels_np[i])
                p_pneu = float(probs[i, 1].cpu().item())
                p_norm = float(probs[i, 0].cpu().item())

                item = {
                    "index": len(tp_candidates) + len(tn_candidates),
                    "true_label": true,
                    "pred_label": pred,
                    "pneumonia_probability": p_pneu,
                    "normal_probability": p_norm,
                    "confidence": max(p_pneu, p_norm),
                    "abs_path": abs_paths[i],
                    "rel_path": rel_paths[i],
                }

                if true == 1 and pred == 1:
                    tp_candidates.append(item)

                if true == 0 and pred == 0:
                    tn_candidates.append(item)

    tp_candidates = rank_candidates(tp_candidates, args.prefer)[:args.num_tp]
    tn_candidates = rank_candidates(tn_candidates, args.prefer)[:args.num_tn]

    print(f"[INFO] Selected {len(tp_candidates)} TP and {len(tn_candidates)} TN.")

    if len(tp_candidates) == 0 or len(tn_candidates) == 0:
        raise RuntimeError("Could not find enough TP or TN samples.")

    target_layer = get_target_layer(model, args.model_name)
    gradcam = GradCAM(model, target_layer)

    ckpt_stem = Path(args.checkpoint).stem
    run_name = args.run_name or ckpt_stem
    out_dir = Path(args.out_dir) / args.model_name / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_items = []

    for group_name, items, target_class in [
        ("TP", tp_candidates, 1),  # visualize pneumonia activation
        ("TN", tn_candidates, 0),  # visualize normal activation
    ]:
        for item in items:
            pil_img = Image.open(item["abs_path"]).convert("RGB")
            rgb_img = np.array(pil_img.resize((args.image_size, args.image_size))).astype(np.float32) / 255.0

            preprocess = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            cam = gradcam(input_tensor, target_class)
            cam_resized = cv2.resize(cam, (args.image_size, args.image_size))
            overlay, heatmap_rgb, original_rgb = overlay_heatmap_on_image(rgb_img, cam_resized, alpha=0.4)

            image_stem = Path(item["rel_path"]).stem
            sample_dir = out_dir / group_name / image_stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            save_image(sample_dir / "original.png", original_rgb)
            save_image(sample_dir / "heatmap.png", heatmap_rgb)
            save_image(sample_dir / "overlay.png", overlay)

            summary = {
                "group": group_name,
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "image_path": item["abs_path"],
                "relative_path": item["rel_path"],
                "true_label": item["true_label"],
                "predicted_label": item["pred_label"],
                "normal_probability": item["normal_probability"],
                "pneumonia_probability": item["pneumonia_probability"],
                "gradcam_target_class": target_class,
                "saved_files": {
                    "original": str((sample_dir / "original.png").as_posix()),
                    "heatmap": str((sample_dir / "heatmap.png").as_posix()),
                    "overlay": str((sample_dir / "overlay.png").as_posix()),
                },
            }

            with (sample_dir / "gradcam_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            item["overlay_img"] = overlay
            selected_items.append(summary)

    gradcam.remove_hooks()

    tp_items_for_panel = []
    tn_items_for_panel = []

    for item in tp_candidates:
        image_stem = Path(item["rel_path"]).stem
        overlay_img = np.array(Image.open(out_dir / "TP" / image_stem / "overlay.png").convert("RGB"))
        item["overlay_img"] = overlay_img
        tp_items_for_panel.append(item)

    for item in tn_candidates:
        image_stem = Path(item["rel_path"]).stem
        overlay_img = np.array(Image.open(out_dir / "TN" / image_stem / "overlay.png").convert("RGB"))
        item["overlay_img"] = overlay_img
        tn_items_for_panel.append(item)

    panel_path = out_dir / "gradcam_tp_tn_panel.png"
    make_panel_figure(tp_items_for_panel, tn_items_for_panel, panel_path, cell_size=args.image_size)

    with (out_dir / "selected_cases.json").open("w", encoding="utf-8") as f:
        json.dump(selected_items, f, indent=2)

    print(f"[INFO] Panel figure saved to: {panel_path}")
    print(f"[INFO] Selected cases JSON saved to: {out_dir / 'selected_cases.json'}")
    print(f"[INFO] Done.")
    print(f"[INFO] TP overlays in: {out_dir / 'TP'}")
    print(f"[INFO] TN overlays in: {out_dir / 'TN'}")


if __name__ == "__main__":
    main()