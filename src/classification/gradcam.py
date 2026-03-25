import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model(model_name: str):
    """
    Build the SAME architecture as in train_resnet.py / evaluate_resnet.py.
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

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_target_layer(model, model_name: str):
    """
    Return the last convolutional layer/block for Grad-CAM.
    """
    model_name = model_name.lower().strip()

    if model_name in ["resnet18", "resnet50"]:
        return model.layer4[-1]

    if model_name == "densenet121":
        return model.features

    raise ValueError(f"Unsupported model_name: {model_name}")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

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
        """
        Returns a normalized Grad-CAM heatmap in [0,1], shape (H, W).
        """
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients          # [B, C, H, W]
        activations = self.activations      # [B, C, H, W]

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)

        cam = cam[0, 0].cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam


def overlay_heatmap_on_image(rgb_img_float, cam, alpha=0.4):
    """
    rgb_img_float: numpy array [H,W,3] in [0,1]
    cam: numpy array [H,W] in [0,1]
    """
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = heatmap * alpha + rgb_img_float * (1 - alpha)
    overlay = overlay / np.maximum(overlay.max(), 1e-8)
    overlay = np.uint8(255 * overlay)

    return overlay, np.uint8(255 * heatmap), np.uint8(255 * rgb_img_float)


def load_image(image_path: str, image_size: int = 224):
    """
    Load image exactly like your training/eval pipeline:
    resize -> tensor -> ImageNet normalization
    """
    pil_img = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    input_tensor = preprocess(pil_img).unsqueeze(0)

    rgb_img = np.array(pil_img.resize((image_size, image_size))).astype(np.float32) / 255.0

    return pil_img, rgb_img, input_tensor


def save_image(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for pneumonia classifier.")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Class to visualize: 0=Normal, 1=Pneumonia. If omitted, uses predicted class.")
    parser.add_argument("--out_dir", type=str, default="results/classification/gradcam")
    parser.add_argument("--run_name", type=str, default=None)
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

    _, rgb_img, input_tensor = load_image(args.image_path, image_size=args.image_size)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_class].item())
        pneumonia_prob = float(probs[1].item())

    target_class = pred_class if args.class_idx is None else int(args.class_idx)

    print(f"[INFO] Predicted class: {pred_class}")
    print(f"[INFO] Predicted class prob: {pred_prob:.6f}")
    print(f"[INFO] Pneumonia prob: {pneumonia_prob:.6f}")
    print(f"[INFO] Grad-CAM target class: {target_class}")

    target_layer = get_target_layer(model, args.model_name)
    gradcam = GradCAM(model, target_layer)

    cam = gradcam(input_tensor, target_class)
    gradcam.remove_hooks()

    cam_resized = cv2.resize(cam, (args.image_size, args.image_size))
    overlay, heatmap_rgb, original_rgb = overlay_heatmap_on_image(rgb_img, cam_resized, alpha=0.4)

    image_stem = Path(args.image_path).stem
    ckpt_stem = Path(args.checkpoint).stem
    run_name = args.run_name or ckpt_stem

    out_dir = Path(args.out_dir) / args.model_name / run_name / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(out_dir / "original.png", original_rgb)
    save_image(out_dir / "heatmap.png", heatmap_rgb)
    save_image(out_dir / "overlay.png", overlay)

    summary = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "image_path": args.image_path,
        "image_size": args.image_size,
        "predicted_class": pred_class,
        "predicted_class_probability": pred_prob,
        "pneumonia_probability": pneumonia_prob,
        "gradcam_target_class": target_class,
        "target_layer": str(target_layer),
        "saved_files": {
            "original": str((out_dir / "original.png").as_posix()),
            "heatmap": str((out_dir / "heatmap.png").as_posix()),
            "overlay": str((out_dir / "overlay.png").as_posix()),
        },
    }

    with (out_dir / "gradcam_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved original image to: {out_dir / 'original.png'}")
    print(f"[INFO] Saved heatmap to: {out_dir / 'heatmap.png'}")
    print(f"[INFO] Saved overlay to: {out_dir / 'overlay.png'}")
    print(f"[INFO] Saved summary to: {out_dir / 'gradcam_summary.json'}")


if __name__ == "__main__":
    main()




"""python src/classification/gradcam.py --model_name resnet50 --checkpoint models/resnet50_FINAL.pth --image_path "C:\Users\houta\Desktop\MA2\PROJ-H419-Pneumonia\data\png\test\02b7f2f0-b8e0-41c8-9cfc-83c69d202c67.png" --class_idx 1"""