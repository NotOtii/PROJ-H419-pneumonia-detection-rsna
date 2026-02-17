import argparse
from pathlib import Path

import pandas as pd


def build_classification_table(labels_csv: Path, images_dir: Path) -> pd.DataFrame:
    """
    Build a table mapping each image to a binary classification label.

    Label definition:
        0 = normal
        1 = pneumonia

    If a patientId has at least one bounding box (Target=1),
    the image is labeled as pneumonia.
    """
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    print(f"[INFO] Reading labels from {labels_csv}")
    df = pd.read_csv(labels_csv)

    if "patientId" not in df.columns or "Target" not in df.columns:
        raise ValueError("CSV must contain 'patientId' and 'Target' columns.")

    # Aggregate per patientId
    grouped = (
        df.groupby("patientId")["Target"]
        .max()
        .reset_index()
        .rename(columns={"Target": "label"})
    )

    print(f"[INFO] Found {len(grouped)} unique patient IDs in labels.")

    image_paths = []
    labels = []

    missing_images = 0

    for _, row in grouped.iterrows():
        patient_id = row["patientId"]
        label = int(row["label"])

        png_path = images_dir / f"{patient_id}.png"

        if png_path.exists():
            image_paths.append(png_path.as_posix())
            labels.append(label)
        else:
            missing_images += 1

    data = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })

    # Summary statistics
    total = len(data)
    positives = int(data["label"].sum())
    negatives = total - positives

    print("[INFO] ----------------------------")
    print(f"[INFO] Final dataset size: {total}")
    print(f"[INFO] Pneumonia cases: {positives}")
    print(f"[INFO] Normal cases: {negatives}")
    print(f"[INFO] Positive ratio: {positives / total:.4f}")
    print(f"[INFO] Missing PNG files skipped: {missing_images}")
    print("[INFO] ----------------------------")

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare classification labels CSV from RSNA dataset."
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to RSNA labels CSV file.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing PNG images.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/classification_labels.csv",
        help="Path to output classification CSV file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    labels_csv = Path(args.labels_csv)
    images_dir = Path(args.images_dir)
    output_csv = Path(args.output_csv)

    df = build_classification_table(labels_csv, images_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[INFO] Classification labels saved to {output_csv}")


if __name__ == "__main__":
    main()

"""
python src\data_src\prepare_classification_labels.py --labels_csv data\raw\stage_1_train_labels.csv --images_dir data\png\train --output_csv data\classification_labels.csv

"""