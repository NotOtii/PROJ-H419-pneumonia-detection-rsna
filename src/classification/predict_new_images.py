"""
predict_new_images.py
---------------------
Run inference on unseen chest X-ray images using a trained classifier.

Loads a saved checkpoint, processes each image in the input directory,
and prints the predicted class (NORMAL / PNEUMONIA) with probabilities.

Usage:
    python src/classification/predict_new_images.py \
        --model_name resnet50 \
        --checkpoint models/resnet50_FINAL.pth \
        --input_dir data/new_images \
        --threshold 0.3
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
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


def get_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_image(model, image_path: Path, device, threshold: float):
    transform = get_transform()

    img = Image.open(image_path).convert("L").convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        prob_pneumonia = float(probs[1].item())

    pred = 1 if prob_pneumonia >= threshold else 0
    label = "PNEUMONIA" if pred == 1 else "NORMAL"

    return {
        "image": str(image_path),
        "prob_normal": float(probs[0].item()),
        "prob_pneumonia": prob_pneumonia,
        "threshold": threshold,
        "prediction": label,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict pneumonia on new chest X-ray images.")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "densenet121"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing new images (.png, .jpg, .jpeg)")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_model(args.model_name).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    input_dir = Path(args.input_dir)
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(input_dir.glob(ext))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"[INFO] Found {len(image_paths)} images")

    for image_path in image_paths:
        result = predict_image(model, image_path, device, args.threshold)
        print("-" * 60)
        print(f"Image: {result['image']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Prob(normal): {result['prob_normal']:.4f}")
        print(f"Prob(pneumonia): {result['prob_pneumonia']:.4f}")
        print(f"Threshold: {result['threshold']:.2f}")


if __name__ == "__main__":
    main()