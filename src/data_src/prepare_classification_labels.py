import argparse
from pathlib import Path

import pandas as pd


def build_classification_table(labels_csv: Path, images_dir: Path) -> pd.DataFrame:
    """
    Build a table mapping each image to a classification label (0 = normal, 1 = pneumonia).

    The RSNA labels CSV contains multiple rows per patientId when there are multiple
    bounding boxes. For classification, we only need a binary label:
        - label = 1 if the patient has at least one pneumonia box (Target = 1)
        - label = 0 otherwise (Target = 0 for all rows)
    """
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    print(f"[INFO] Reading labels from {labels_csv}")
    df = pd.read_csv(labels_csv)

    if "patientId" not in df.columns or "Target" not in df.columns:
        raise ValueError("CSV must contain 'patientId' and 'Target' columns.")

    # Aggregate: one row per patientId with label = max(Target)
    # (0 if all rows are 0, 1 if at least one row is 1)
    grouped = (
        df.groupby("patientId")["Target"]
        .max()
        .reset_index()
        .rename(columns={"Target": "label"})
    )

    print(f"[INFO] Found {len(grouped)} unique patient IDs in labels.")

    # Build image paths and keep only those that exist as PNG files
    image_paths = []
    labels = []

    for _, row in grouped.iterrows():
        patient_id = row["patientId"]
        label = int(row["label"])

        png_path = images_dir / f"{patient_id}.png"

        if png_path.exists():
            image_paths.append(png_path.as_posix())
            labels.append(label)
        else:
            # It is possible that some IDs are missing if not all DICOMs were converted
            # We silently skip those.
            continue

    data = pd.DataFrame({"image_path": image_paths, "label": labels})
    print(f"[INFO] Kept {len(data)} images with existing PNG files.")

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a classification labels CSV file from RSNA labels and PNG images.\n"
            "Each row links an image path to a binary label (0 = normal, 1 = pneumonia)."
        )
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to RSNA labels CSV file (e.g., data/raw/stage_1_train_labels.csv).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Folder containing PNG images (e.g., data/png/train).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/classification_labels.csv",
        help="Path to the output CSV file.",
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
Command to run in the terminal to build the classification labels table:
------------------------------------------------------------------------
Run this from the root of the project (PROJ-H419-Pneumonia):

python src/data_src/prepare_classification_labels.py --labels_csv data/raw/stage_1_train_labels.csv --images_dir data/png/train --output_csv data/classification_labels.csv


------------------------------------------------------------
SCRIPT EXPLANATION: prepare_classification_labels.py
------------------------------------------------------------

Goal
----
This script creates a CSV file that links each X-ray image (PNG) to a binary
classification label:
    - 0 = normal (no pneumonia)
    - 1 = pneumonia present

It uses the official RSNA labels file (stage_1_train_labels.csv), where:
    - 'patientId' identifies the image
    - 'Target' is 0 or 1
    - multiple rows may exist per patientId (multiple bounding boxes)

For classification, we only need to know:
    "Does this image show pneumonia or not?"

How it works
------------

1. Read the RSNA labels CSV
   - The script loads the CSV file using pandas.
   - It checks that the columns 'patientId' and 'Target' are present.

2. Aggregate labels per patientId
   - Each patientId can have multiple rows (different boxes).
   - We compute:
        label = max(Target) per patientId
     This gives:
        • label = 1 if the image has at least one pneumonia annotation
        • label = 0 otherwise

3. Associate each patientId with a PNG image
   - After DICOM → PNG conversion, each image has a file name:
        <patientId>.png
   - The script constructs the path:
        images_dir / "<patientId>.png"
   - Only images that actually exist on disk are kept.

4. Save the final table
   - The resulting CSV has two columns:
        image_path,label
     Example:
        data/png/train/0004cfab-14fd-4e49-80ba-63a80b6bddd6.png,1
        data/png/train/000a312494fe8f.png,0
   - This CSV will be used by the classification DataLoader (e.g., ResNet training).

Why this is necessary
---------------------
The RSNA dataset does not directly provide "normal / pneumonia" folders.
Labels are stored in a separate CSV file, and bounding boxes are provided
for localization.

For a classification task, we must:
    • compress the box-level annotations into a single binary label per image
    • link that label to the corresponding PNG file path

This script automates that process and produces a clean, reusable labels file.
------------------------------------------------------------
"""
